import numpy as np


class FakeEnv(object):
    """
    FakeEnv.transition_model: forward prediction model
    FakeEnv._state: internal state, np.ndarray w/ shape (num_paths, obs_size)
    """
    def __init__(self, transition_model, obs_space, act_space):
        self.transition_model = transition_model
        self.observation_space = obs_space
        self.action_space = act_space
        self._state = None

    def step(self, action):
        """
        This is designed to emulate an OpenAI gym environment interface.
        :param action: np.ndarray. If action is 1-D and self._state is 2-D, the same action
                       will be applied to step forward all states.
        :return: next_state (np.ndarray) , reward (np.ndarray), done (np.ndarray), info (None)
        """
        if self._state is None:
            raise RuntimeError("FakeEnv state has not been set.")
        if action.ndim < 2:
            action = action.reshape(1, -1)
            action = np.repeat(action, self._state.shape[0], axis=0)
        elif action.ndim > 2:
            raise ValueError("action should be scalar, 1-D or 2-D")

        model_inputs = np.concatenate([self._state, action], axis=-1)
        reward_delta = self.transition_model.sample_next(model_inputs)
        next_state = self._state + reward_delta[:, 1:]
        reward = reward_delta[:, 0]
        done = np.zeros(reward.shape)
        self.set_state(next_state)

        return next_state, reward, done, None

    def set_state(self, state):
        if state.ndim == 1:
            self._state = state.reshape(1, -1)
        elif state.ndim == 2:
            self._state = state
        else:
            raise ValueError("state must be 1 or 2-D.")

    def fit_model(self, inputs, labels, holdout_ratio=None):
        self.transition_model.fit(inputs, labels, holdout_ratio)
