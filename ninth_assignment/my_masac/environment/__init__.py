import numpy as np
from argparse import Namespace
from my_masac.environment.mpe import MPE_Env
from my_masac.utils.utils import space2shape
from my_masac.environment.mpe import MultiAgentEnvWrapper
# referenced from openai/baselines
class AlreadySteppingError(Exception):
    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)


class DummyVecMultiAgentEnv:#(VecEnv)
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    Parameters:
        env_fns – environment function.
    """

    def __init__(self, env_fns, env_seed):
        self.waiting = False
        self.closed = False
        self.envs = [fn(env_seed=env_seed + inx_env) for inx_env, fn in enumerate(env_fns)]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.closed = False
        # VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.env_info = env.env_info
        self.groups_info = env.groups_info
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.state_space = env.state_space  # Type: Box
        self.buf_state = [np.zeros(space2shape(self.state_space)) for _ in range(self.num_envs)]
        self.buf_obs = [{} for _ in range(self.num_envs)]
        self.buf_avail_actions = [{} for _ in range(self.num_envs)]
        self.buf_info = [{} for _ in range(self.num_envs)]

        self.actions = None
        self.max_episode_steps = env.max_episode_steps

    def reset(self):
        """Reset the vectorized environments."""
        for e in range(self.num_envs):
            self.buf_obs[e], self.buf_info[e] = self.envs[e].reset()
            self.buf_state[e] = self.buf_info[e]['state']
            self.buf_avail_actions[e] = self.buf_info[e]['avail_actions']
        return self.buf_obs.copy(), self.buf_info.copy()

    def step_async(self, actions):
        """Sends asynchronous step commands to each subprocess with the specified actions."""
        if self.waiting:
            raise AlreadySteppingError
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass
        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                actions, self.num_envs)
            self.actions = [actions]
        self.waiting = True

    def step_wait(self):
        """
        Waits for the completion of asynchronous step operations and updates internal buffers with the received results.
        """
        if not self.waiting:
            raise NotSteppingError

        rew_dict = [{} for _ in self.envs]
        terminated_dict = [{} for _ in self.envs]
        truncated = [False for _ in self.envs]
        for e in range(self.num_envs):
            action_n = self.actions[e]
            self.buf_obs[e], rew_dict[e], terminated_dict[e], truncated[e], self.buf_info[e] = self.envs[e].step(action_n)
            self.buf_avail_actions[e] = self.buf_info[e]['avail_actions']
            self.buf_state[e] = self.buf_info[e]['state']
            if all(terminated_dict[e].values()) or truncated[e]:
                obs_reset_dict, info_reset = self.envs[e].reset()
                self.buf_info[e]["reset_obs"] = obs_reset_dict
                self.buf_info[e]["reset_avail_actions"] = info_reset['avail_actions']
                self.buf_info[e]["reset_state"] = info_reset['state']
        self.waiting = False
        return self.buf_obs.copy(), rew_dict, terminated_dict, truncated, self.buf_info.copy()

    def close_extras(self):
        """Closes the communication with subprocesses and joins the subprocesses."""
        self.closed = True
        for env in self.envs:
            try:
                env.close()
            except:
                pass

    def render(self, mode):
        return [env.render(mode) for env in self.envs]

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def close(self):
        if self.closed is True:
            return
        self.close_extras()
        self.closed = True


def make_envs(config: Namespace):
    def make(env_seed = None):
        config.env_seed = env_seed
        return MultiAgentEnvWrapper(MPE_Env(config))#XuanCeMultiAgentEnvWrapper()
    env_fn = [make for _ in range(config.parallels)]   
    return DummyVecMultiAgentEnv(env_fn, config.env_seed)
    
