import numpy as np

from optimistic_exploration import get_optimistic_exploration_action


class ModelPathCollector(object):
    def __init__(self, fake_env, env_buffer):
        self.fake_env = fake_env
        self.env_buffer = env_buffer

    def collect_new_paths(
            self,
            policy,
            num_steps,
            rollout_batch_size,
            optimistic_exploration=False,
            optimistic_exploration_kwargs={},
    ):
        initial_data = self.env_buffer.random_batch(rollout_batch_size)
        start_obs = initial_data['observations']
        paths = model_rollout(start_obs,
                              self.fake_env,
                              policy,
                              num_steps,
                              optimistic_exploration,
                              optimistic_exploration_kwargs)
        return paths


def model_rollout(
        start_state,
        fake_env,
        agent,
        num_steps,
        optimistic_exploration,
        optimistic_exploration_kwargs,
):
    batch_size, obs_size = start_state.shape
    observations = np.empty((num_steps, batch_size, obs_size))
    next_observations = np.empty((num_steps, batch_size, obs_size))
    actions = np.empty((num_steps, batch_size) + fake_env.action_space.shape)
    rewards = np.empty((num_steps, batch_size))
    terminals = np.empty((num_steps, batch_size))

    agent.reset()
    batch_o = start_state
    batch_a = None
    batch_next_o = None
    path_length = 0

    for t in range(num_steps):
        if not optimistic_exploration:
            batch_a, _ = agent.get_actions(batch_o)
        else:
            # get_optimistic_action doesn't support batching
            batch_a = [get_optimistic_exploration_action(
                o, **optimistic_exploration_kwargs
            )[0] for o in batch_o]
            batch_a = np.stack([batch_a])
            # a, _ = get_optimistic_exploration_action(
            #     o, **optimistic_exploration_kwargs)

        fake_env.set_state(batch_o)
        batch_next_o, batch_r, batch_d, _ = fake_env.step(batch_a)

        observations[t] = batch_o
        actions[t] = batch_a
        rewards[t] = batch_r
        next_observations[t] = batch_next_o
        terminals[t] = batch_d
        path_length += 1
        batch_o = batch_next_o

    paths = [
        dict(
            observations=observations[:, path_idx],
            actions=actions[:, path_idx],
            rewards=rewards[:, path_idx],
            next_observations=next_observations[:, path_idx],
            terminals=terminals[:, path_idx],
            env_infos=[None] * path_length,
            agent_infos=[None] * path_length
        ) for path_idx in range(batch_size)
    ]
    return paths

