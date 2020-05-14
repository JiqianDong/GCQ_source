from environment import Env
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from gym.spaces.box import Box
from gym.spaces import Discrete
from gym.spaces.dict import Dict

class MergeEnv(Env):

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.
        """
        N = self.net_params.additional_params['num_vehicles']
        F = 3 + self.net_params.additional_params['highway_lanes']\
            + self.n_unique_intentions

        states = Box(low=-np.inf, high=np.inf, shape=(N,F), dtype=np.float32)
        adjacency = Box(low=0, high=1, shape = (N,N), dtype=np.int32)
        mask = Box(low=0, high=1, shape = (N,), dtype=np.int32)

        return Dict({'states':states,'adjacency':adjacency,'mask':mask})

    @property
    def action_space(self):

        return Box(low=0,high=1,shape(N,),dtype=np.int32)
        # return Discrete(3)


    def get_state(self):
        """construct a graph for each time step
        """
        N = self.net_params.additional_params['num_vehicles']
        num_cav = self.net_params.additional_params['num_cav']
        num_hv = self.net_params.additional_params['num_hv']

        num_lanes = self.net_params.additional_params['highway_lanes']

        ids = self.k.vehicle.get_ids()
        rl_ids = self.k.vehicle.get_rl_ids()
        human_ids = self.k.vehicle.get_human_ids()

        # assert len(ids) != len(human_ids) + len(rl_ids)


        states = np.zeros([N,3+num_lanes+self.n_unique_intentions])
        adjacency = np.zeros([N,N])
        mask = np.zeros(N)

        if rl_ids: ## when there is rl_vehicles in the scenario

            ids = human_ids + rl_ids

            # numerical data (speed, location)
            speeds = np.array(self.k.vehicle.get_speed(ids)).reshape(-1,1)
            positions = np.array([self.k.vehicle.get_absolute_position(i) for i in ids])

            # categorical data  1 hot encoding: (lane location, intention)
            lanes_column = np.array(self.k.vehicle.get_lane(ids))
            lanes = np.zeros([len(ids),num_lanes])
            lanes[np.arange(len(ids)),lanes_column] = 1

            # intention encoding
            types_column = np.array([self.intention_dict[self.k.vehicle.get_type(i)] for i in ids])
            intention = np.zeros([len(ids), self.n_unique_intentions])
            intention[np.arange(len(ids)),types_column] = 1

            observed_states = np.c_[positions,speeds,lanes,intention]

            # assemble into the NxF states matrix
            states[:len(human_ids),:] = observed_states[:len(human_ids),:]
            states[num_hv:num_hv+len(rl_ids),:] = observed_states[len(human_ids):,:]


            # construct the adjacency matrix
            dist_matrix = euclidean_distances(positions)
            adjacency_small = np.zeros_like(dist_matrix)
            adjacency_small[dist_matrix<20] = 1
            adjacency_small[-len(rl_ids):,-len(rl_ids):] = 1

            # assemble into the NxN adjacency matrix
            adjacency[:len(human_ids),:len(human_ids)] = adjacency_small[:len(human_ids),:len(human_ids)]
            adjacency[num_hv:num_hv+len(rl_ids),:len(human_ids)] = adjacency_small[len(human_ids):,:len(human_ids)]
            adjacency[:len(human_ids),num_hv:num_hv+len(rl_ids)] = adjacency_small[:len(human_ids),len(human_ids):]
            adjacency[num_hv:num_hv+len(rl_ids),num_hv:num_hv+len(rl_ids)] = adjacency_small[len(human_ids):,len(human_ids):]

            # construct the mask
            mask[num_hv:num_hv+len(rl_ids)] = np.ones(len(rl_ids))

            # print(adjacency)
            # print(states)
            # print(mask)

        return {"states":states, "adjacency":adjacency, "mask":mask}

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