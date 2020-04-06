from environment import Env
import numpy as np

class MergeEnv(Env):

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.
        """
        return Box(low=-np.inf, high=np.inf, shape=(self.net_params['addtional_params']['num_vehicles'],4), dtype=np.float32)

    @property
    def action_space(self):
        return Discrete(3)


    def get_state(self):
        N = self.net_params.additional_params['num_vehicles']
        states = np.zeros([N,4])
        ids = self.k.vehicle.get_ids()
        # if ids:
        #     speeds = np.array(self.k.vehicle.get_speed(ids)).reshape(-1,1)
        #     xs = np.array([self.k.vehicle.get_x_by_id(i) for i in ids]).reshape(-1,1)
        #     lanes = np.array(self.k.vehicle.get_lane(ids)).reshape(-1,1)
        #     types = np.array([self.intention[self.k.vehicle.get_type(i)] for i in ids]).reshape(-1,1)

        return states

    def compute_reward(self,rl_actions,**kwargs):
        return 0
