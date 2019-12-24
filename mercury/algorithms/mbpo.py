import abc
from collections import OrderedDict

from utils.logging import logger
import utils.eval_util as eval_util
from utils.rng import get_global_pkg_rng_state
import utils.pytorch_util as ptu
from utils.env_utils import get_dim

import gtimer as gt
from replay_buffer import ReplayBuffer
from path_collector import MdpPathCollector, RemoteMdpPathCollector
from mercury.models.model_path_collector import ModelPathCollector
from tqdm import trange

import ray
import numpy as np


class MBPOAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_data_collector: MdpPathCollector,
            remote_eval_data_collector: RemoteMdpPathCollector,
            replay_buffer: ReplayBuffer,
            fake_env,
            batch_size,
            max_path_length,
            num_epochs,
            num_train_steps_per_epoch,
            num_agent_updates_per_train_step,
            model_train_freq,
            holdout_ratio,
            rollout_length_schedule,
            rollout_batch_size,
            num_rollouts_retained,
            num_eval_steps_per_epoch,
            min_num_steps_before_training=0,
            optimistic_exp_hp=None,
    ):
        super().__init__()

        """
        The class state which should not mutate
        """
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.min_num_steps_before_training = min_num_steps_before_training
        self.optimistic_exp_hp = optimistic_exp_hp

        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_train_steps_per_epoch = num_train_steps_per_epoch
        self.num_agent_updates_per_train_step = num_agent_updates_per_train_step

        # model settings
        self.model_train_freq = model_train_freq
        self.holdout_ratio = holdout_ratio
        self.rollout_batch_size = rollout_batch_size
        self.num_rollouts_retained = num_rollouts_retained
        self.rollout_length_schedule = rollout_length_schedule
        self.rollout_length = rollout_length_schedule[-2]

        """
        The class mutable state
        """
        self._start_epoch = 0

        """
        This class sets up the main training loop, so it needs reference to other
        high level objects in the algorithm

        But these high level object maintains their own states
        and has their own responsibilities in saving and restoring their state for checkpointing
        """
        self.trainer = trainer
        self.fake_env = fake_env

        self.expl_data_collector = exploration_data_collector
        self.remote_eval_data_collector = remote_eval_data_collector

        self.env_buffer = replay_buffer
        self.model_buffer = self._allocate_model_buffer()
        self.model_data_collector = ModelPathCollector(self.fake_env, self.env_buffer)

    @property
    def _max_model_buffer_size(self):
        return self.rollout_batch_size * self.rollout_length * self.num_rollouts_retained

    def _allocate_model_buffer(self):
        empty_buffer = DynamicReplayBuffer(
            self._max_model_buffer_size,
            self.env_buffer._ob_space,
            self.env_buffer._action_space
        )
        return empty_buffer

    def _fit_transition_model(self):
        obs = self.env_buffer._observations[:self.env_buffer._size]
        actions = self.env_buffer._actions[:self.env_buffer._size]
        rewards = self.env_buffer._rewards[:self.env_buffer._size]
        next_obs = self.env_buffer._next_obs[:self.env_buffer._size]
        train_inputs = np.concatenate([obs, actions])
        delta_obs = next_obs - obs
        train_labels = np.concatenate([rewards, delta_obs], axis=-1)
        self.fake_env.fit_model(
            train_inputs,
            train_labels,
            holdout_ratio=self.holdout_ratio
        )

    def _update_rollout_length(self, epoch):
        start_epoch, end_epoch, min_length, max_length = self.rollout_length_schedule
        length_range = max_length - min_length
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        progress = max(0, progress)
        progress = min(1, progress)
        self.rollout_length = int(min_length + progress * length_range)

    def _train(self):
        # Fill the replay buffer to a minimum before training starts
        if self.min_num_steps_before_training > self.env_buffer.num_steps_can_sample():
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.trainer.policy,
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.env_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                trange(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):

            # To evaluate the policy remotely,
            # we're shipping the policy params to the remote evaluator
            # This can be made more efficient
            # But this is currently extremely cheap due to small network size
            pol_state_dict = ptu.state_dict_cpu(self.trainer.policy)

            remote_eval_obj_id = self.remote_eval_data_collector.async_collect_new_paths.remote(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,

                deterministic_pol=True,
                pol_state_dict=pol_state_dict)

            gt.stamp('remote evaluation submit')

            self._update_rollout_length(epoch)
            for env_step in range(self.num_train_steps_per_epoch):
                if env_step % self.model_train_freq == 0:
                    self._fit_transition_model()
                    rollout_paths = self.model_data_collector.collect_new_paths(
                        self.trainer.policy,
                        num_steps=self.rollout_length,
                        rollout_batch_size=self.rollout_batch_size,
                        optimistic_exploration=self.optimistic_exp_hp['should_use'],
                        optimistic_exploration_kwargs=dict(
                            policy=self.trainer.policy,
                            qfs=[self.trainer.qf1, self.trainer.qf2],
                            hyper_params=self.optimistic_exp_hp
                        )
                    )
                    self.model_buffer.reallocate(self._max_model_buffer_size)
                    self.model_buffer.add_paths(rollout_paths)

                self.expl_data_collector.collect_new_paths(
                    self.trainer.policy,
                    self.max_path_length,
                    num_steps=1,
                    discard_incomplete_paths=False,
                    optimistic_exploration=self.optimistic_exp_hp['should_use'],
                    optimistic_exploration_kwargs=dict(
                        policy=self.trainer.policy,
                        qfs=[self.trainer.qf1, self.trainer.qf2],
                        hyper_params=self.optimistic_exp_hp
                    )
                )

                for _ in range(self.num_agent_updates_per_train_step):
                    train_data = self.model_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)

            self.env_buffer.add_paths(self.expl_data_collector.get_epoch_paths())
            gt.stamp('data storing', unique=False)

            # Wait for eval to finish
            ray.get([remote_eval_obj_id])
            gt.stamp('remote evaluation wait')

            self._end_epoch(epoch)

            # for _ in range(self.num_train_loops_per_epoch):
            #     new_expl_paths = self.expl_data_collector.collect_new_paths(
            #         self.trainer.policy,
            #         self.max_path_length,
            #         self.num_expl_steps_per_train_loop,
            #         discard_incomplete_paths=False,
            #
            #         optimistic_exploration=self.optimistic_exp_hp['should_use'],
            #         optimistic_exploration_kwargs=dict(
            #             policy=self.trainer.policy,
            #             qfs=[self.trainer.qf1, self.trainer.qf2],
            #             hyper_params=self.optimistic_exp_hp
            #         )
            #     )
            #     gt.stamp('exploration sampling', unique=False)
            #
            #     self.env_buffer.add_paths(new_expl_paths)
            #     gt.stamp('data storing', unique=False)
            #
            #     for _ in range(self.num_trains_per_train_loop):
            #         train_data = self.env_buffer.random_batch(
            #             self.batch_size)
            #         self.trainer.train(train_data)
            #     gt.stamp('training', unique=False)

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _end_epoch(self, epoch):
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        ray.get([self.remote_eval_data_collector.end_epoch.remote(epoch)])

        self.env_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        # We can only save the state of the program
        # after we call end epoch on all objects with internal state.
        # This is so that restoring from the saved state will
        # lead to identical result as if the program was left running.
        if epoch > 0:
            snapshot = self._get_snapshot(epoch)
            logger.save_itr_params(epoch, snapshot)
            gt.stamp('saving')

        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)

        write_header = True if epoch == 0 else False
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_header)

    def _get_snapshot(self, epoch):
        snapshot = dict(
            trainer=self.trainer.get_snapshot(),
            exploration=self.expl_data_collector.get_snapshot(),
            evaluation_remote=ray.get(
                self.remote_eval_data_collector.get_snapshot.remote()),
            evaluation_remote_rng_state=ray.get(
                self.remote_eval_data_collector.get_global_pkg_rng_state.remote()
            ),
            replay_buffer=self.env_buffer.get_snapshot()
        )

        # What epoch indicates is that at the end of this epoch,
        # The state of the program is snapshot
        # Not to be consfused with at the beginning of the epoch
        snapshot['epoch'] = epoch

        # Save the state of various rng
        snapshot['global_pkg_rng_state'] = get_global_pkg_rng_state()

        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.env_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Remote Evaluation
        """
        logger.record_dict(
            ray.get(self.remote_eval_data_collector.get_diagnostics.remote()),
            prefix='remote_evaluation/',
        )
        remote_eval_paths = ray.get(
            self.remote_eval_data_collector.get_epoch_paths.remote())
        logger.record_dict(
            eval_util.get_generic_path_information(remote_eval_paths),
            prefix="remote_evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging')

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class DynamicReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            ob_space,
            action_space,
    ):
        super().__init__(max_replay_buffer_size, ob_space, action_space)
        self.ob_dim = get_dim(ob_space)
        self.ac_dim = get_dim(action_space)

    def reallocate(self, max_replay_buffer_size):
        obs = self._observations[:self._size]
        actions = self._actions[:self._size]
        rewards = self._rewards[:self._size]
        next_obs = self._next_obs[:self._size]
        terminals = self._terminals[:self._size]

        self._observations = np.zeros((max_replay_buffer_size, self.ob_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, self.ob_dim))
        self._actions = np.zeros((max_replay_buffer_size, self.ac_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        n = min(max_replay_buffer_size, self._size)
        self._observations[:n] = obs[-n:]
        self._next_obs[:n] = next_obs[-n:]
        self._actions[:n] = actions[-n:]
        self._rewards[:n] = rewards[-n:]
        self._terminals[:n] = terminals[-n:]

        if self._size > max_replay_buffer_size:
            self._top = 0
        self._size = n
        self._max_replay_buffer_size = max_replay_buffer_size
