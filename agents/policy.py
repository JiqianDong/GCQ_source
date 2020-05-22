import numpy as np

class Policy(object):
    """Abstract base class for all implemented policies.

    Each policy helps with selection of action to take on an environment.

    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:

    - `select_action`

    # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy

        # Returns
            Configuration as dict
        """
        return {}


class greedy_q_policy(Policy):
    def select_action(self,q_vals):
        action = None
        mask = np.any(q_vals, axis=1)
        if mask.sum() > 0:
            action = q_vals[mask,:].argmax(1)
        return action

class random_obs_policy(Policy):
    def select_action(self,observation):
        action = None
        _,_,mask = observation
        num_agent = mask.sum().astype(int)
        if num_agent>0:
            action = np.random.choice(np.arange(3),num_agent)
        return action

class eps_greedy_q_policy(Policy):
    def __init__(self, eps=.1):
        super(eps_greedy_q_policy, self).__init__()
        self.eps = eps

    def select_action(self,q_vals):
        action = None
        mask = np.any(q_vals, axis=1)
        num_agent = mask.sum().astype(int)
        if num_agent>0:
            if np.random.uniform() < self.eps:  # choose random action
                action = np.random.choice(np.arange(3),num_agent)
            else:
                action = q_vals[mask,:].argmax(1)
        return action