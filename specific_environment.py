from environment import Env
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time

from gym.spaces.box import Box
from gym.spaces import Discrete
from gym.spaces import Tuple

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

        return Tuple(states,adjacency,mask)

    @property
    def action_space(self):
        N = self.net_params.additional_params['num_vehicles']
        return Box(low=0,high=1,shape=(N,),dtype=np.int32)
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
            self.observed_cavs = rl_ids

        return states, adjacency, mask

    def compute_reward(self,rl_actions,**kwargs):
        unit = 1
        # reward for system speed
        # all_speed = np.array(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))

        # reward for satisfying intention ---- only a big instant reward
        intention_reward = kwargs['num_full_filled'] * unit + kwargs['num_half_filled'] * unit * 0.5
        # if intention_reward>0:
        #     print('current num_full_filled: ',kwargs['num_full_filled'])
        #     print('current num_half_filled: ',kwargs['num_half_filled'])


        # penalty for drastic lane changing behavors
        # total_lane_change_penalty = 0
        # for veh_id in rl_veh_ids:
        #     if self.time_counter - self.k.vehicle.get_last_lc(veh_id)<20:
        #         print("drastic change",veh_id)
        #         print(self.time_counter)
        #         print(self.k.vehicle.get_last_lc(veh_id))
        #         # time.sleep(1)
        #         total_lane_change_penalty -= unit

        # penalty for crashing
        total_crash_penalty = 0
        crash_ids = kwargs["fail"]
        total_crash_penalty = len(crash_ids) * unit
        # if crash_ids:
        #     # time.sleep(1)
        #     print(crash_ids,total_crash_penalty)


        return intention_reward - total_crash_penalty

    def apply_rl_actions(self, rl_actions=None):
        if isinstance(rl_actions,np.ndarray):
            # print(rl_actions.shape)
            # rl_actions = rl_actions.reshape((self.net_params.additional_params['num_cav'],3))
            rl_actions -=1
            rl_ids = self.observed_cavs

            self.k.vehicle.apply_lane_change(rl_ids, rl_actions,2)
        return None

    def check_full_fill(self):
        rl_veh_ids = self.k.vehicle.get_rl_ids()
        num_full_filled = 0
        num_half_filled = 0
        for rl_id in rl_veh_ids:
            if rl_id not in self.exited_vehicles:
                current_edge = self.k.vehicle.get_edge(rl_id)
                if current_edge in self.terminal_edges:
                    self.exited_vehicles.append(rl_id)
                    veh_type = self.k.vehicle.get_type(rl_id)

                    # check if satisfy the intention

                    if self.n_unique_intentions == 3: # specific merge
                        if (veh_type == 'merge_0' and current_edge == 'off_ramp_0') \
                            or (veh_type == 'merge_1' and current_edge == 'off_ramp_1'):
                            num_full_filled += 1
                            print('satisfied: ', rl_id)

                    elif self.n_unique_intentions == 2: # nearest merge
                        num_full_filled += (current_edge == 'off_ramp_0')*1
                        num_half_filled += (current_edge == 'off_ramp_1')*1

                    else:
                        raise Exception("unknown num of unique n_unique_intentions")
        return num_full_filled,num_half_filled