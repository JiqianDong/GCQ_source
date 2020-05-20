from rl.policy import Policy
import numpy as np

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