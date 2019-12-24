import gym
import numpy as np
import torch

from replay_buffer import ReplayBuffer

from mercury.models.pytorch.bnn import PytorchBNN
from mercury.models.fake_env import FakeEnv
from mercury.models.model_path_collector import ModelPathCollector


torch.set_default_tensor_type(torch.cuda.FloatTensor)


class RandomPolicy(torch.nn.Module):
    def __init__(self, action_space):
        self.action_space = action_space

    def forward(self):
        return self.action_space.sample()

    def get_actions(self, obs):
        actions = [self.action_space.sample() for _ in obs]
        return np.stack(actions), None

    def reset(self):
        pass


env = gym.make("Hopper-v2")
env_buffer = ReplayBuffer(int(1e6), env.observation_space, env.action_space)
model_buffer = ReplayBuffer(int(1e6), env.observation_space, env.action_space)
obs_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

o = env.reset()
for _ in range(4096):
    a = env.action_space.sample()
    next_o, r, d, _ = env.step(a)
    env_buffer.add_sample(
        observation=o,
        action=a,
        reward=r,
        next_observation=next_o,
        terminal=d,
        env_info=None
    )
    if d:
        o = env.reset()
    else:
        o = next_o

obs = env_buffer._observations[:env_buffer._size]
actions = env_buffer._actions[:env_buffer._size]
rewards = env_buffer._rewards[:env_buffer._size]
next_obs = env_buffer._next_obs[:env_buffer._size]

nn_ensemble = PytorchBNN(
    input_shape=torch.Size([obs_size + action_size]),
    label_shape=torch.Size([1 + obs_size]),
    hidden_width=200,
    hidden_depth=4,
    ensemble_size=7,
    minibatch_size=256,
    lr=1e-3,
    logvar_penalty_coeff=1e-2,
    max_epochs_since_update=5,
)

train_inputs = np.concatenate([obs, actions], axis=-1)
delta_obs = next_obs - obs
train_labels = np.concatenate([rewards, delta_obs], axis=-1)
nn_ensemble.fit(train_inputs, train_labels, holdout_ratio=0.2, verbose=True)

fake_env = FakeEnv(nn_ensemble, env.observation_space, env.action_space)
model_data_collector = ModelPathCollector(fake_env, env_buffer)
model_paths = model_data_collector.collect_new_paths(
    policy=RandomPolicy(env.action_space),
    num_steps=4,
    rollout_batch_size=256
)
model_buffer.add_paths(model_paths)
print(model_buffer.random_batch(32))
