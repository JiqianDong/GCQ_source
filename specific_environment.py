from environment import Env
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from gym.spaces import Space
from gym.spaces.box import Box
from gym.spaces import Discrete
from gym.spaces.tuple import Tuple

class MergeEnv(Env):

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.
        """
        ids = Space()
        info = Box(low=-np.inf, high=np.inf, shape=(None,
            # self.net_params['addtional_params']['num_vehicles'],
            3 + self.net_params.additional_params['highway_lanes']\
            + self.n_unique_intentions), dtype=np.float32)
        adjacency = Box(low=-np.inf, high=np.inf,dtype=np.float32)
        return Tuple(ids,info,adjacency)

    @property
    def action_space(self):
        return
        # return Discrete(3)


    def get_state(self):
        """construct a graph for each time step
        """
        N = self.net_params.additional_params['num_vehicles']
        num_lanes = self.net_params.additional_params['highway_lanes']

        ids = self.k.vehicle.get_ids()
        rl_ids = self.k.vehicle.get_rl_ids()
        human_ids = self.k.vehicle.get_human_ids()
        # assert len(ids) != len(human_ids) + len(rl_ids)

        ids = human_ids + rl_ids

        states = None
        dist_matrix = None

        if rl_ids: ## when there is rl_vehicles in the scenario

            states = np.zeros([len(ids),3+num_lanes+self.n_unique_intentions])

            # numerical data (speed, location)
            speeds = np.array(self.k.vehicle.get_speed(ids)).reshape(-1,1)
            positions = np.array([self.k.vehicle.get_absolute_position(i) for i in ids])

            # categorical data  1 hot encoding: (lane location, intention)
            lanes_column = np.array(self.k.vehicle.get_lane(ids))
            lanes = np.zeros([len(ids),num_lanes])
            lanes[np.arange(len(ids)),lanes_column] = 1

            types_column = np.array([self.intention_dict[self.k.vehicle.get_type(i)] for i in ids])
            intention = np.zeros([len(ids), self.n_unique_intentions])
            intention[np.arange(len(ids)),types_column] = 1

            states[:len(ids),:] = np.c_[positions,speeds,lanes,intention]

            # construct the adjacency matrix "non weighted"
            dist_matrix = euclidean_distances(positions)

            adjacency = np.zeros_like(dist_matrix)

            adjacency[dist_matrix<10] = 1
            adjacency[-len(rl_ids):,-len(rl_ids):] = 1

        return states,dist_matrix,len(rl_ids)

    def compute_reward(self,rl_actions,**kwargs):
        crash_ids = kwargs["fail"]
        if crash_ids:
            print("crash_ids: ",crash_ids)
            return -1 * len(crash_ids)
        return 1

    def apply_rl_actions(self, rl_actions=None):
        if isinstance(rl_actions,np.ndarray):
            rl_actions = np.array(rl_actions)
            rl_actions -=1
            rl_ids = self.k.vehicle.get_rl_ids()
            self.k.vehicle.apply_lane_change(rl_ids, rl_actions)
        return None