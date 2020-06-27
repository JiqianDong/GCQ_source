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

        # filter the ones on the ramps
        rl_ids = [id_ for id_ in rl_ids if not self.k.vehicle.get_edge(id_).startswith('off_ramp')]
        rl_ids = sorted(rl_ids)

        human_ids = sorted(self.k.vehicle.get_human_ids())

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

            self.observed_cavs = rl_ids
            self.observed_all_vehs = ids

        return states, adjacency, mask

    def compute_reward(self,rl_actions,**kwargs):
        # w_intention = 10
        w_intention = 1
        w_speed = 1
        w_p_lane_change = 1
        w_p_crash = 5

        unit = 1

        # reward for system speed: mean(speed/max_speed) for every vehicle
        speed_reward = 0
        intention_reward = 0

        if self.observed_cavs:
            # all_speed = np.array(self.k.vehicle.get_speed(self.observed_all_vehs))
            # max_speed = np.array([self.env_params.additional_params['max_hv_speed']]*(len(self.observed_all_vehs) - len(self.observed_cavs))\
            #                     +[self.env_params.additional_params['max_cav_speed']]*len(self.observed_cavs))

            all_speed = np.array(self.k.vehicle.get_speed(self.observed_cavs))
            max_speed = self.env_params.additional_params['max_cav_speed']
            speed_reward = np.mean(all_speed/max_speed)
            # print(speed_reward)

        ###### reward for satisfying intention ---- only a big instant reward
        # intention_reward = kwargs['num_full_filled'] * unit + kwargs['num_half_filled'] * unit * 0.5

            for cav_id in self.observed_cavs:
                cav_lane = self.k.vehicle.get_lane(cav_id)

                # print(cav_id,x,cav_lane)
                if cav_lane == 0:
                    # print('here')
                    x = self.k.vehicle.get_x_by_id(cav_id)
                    cav_edge = self.k.vehicle.get_edge(cav_id)
                    cav_type = self.k.vehicle.get_type(cav_id)
                    # total_length = self.net_params.additional_params['highway_length']
                    if (cav_type == 'merge_0' and cav_edge == 'highway_0'):
                        val = (self.net_params.additional_params['off_ramps_pos'][0] - x)/self.net_params.additional_params['off_ramps_pos'][0]
                        intention_reward += val
                        # print('1: ',cav_id,val)
                    elif (cav_type == 'merge_1' and cav_edge == 'highway_1'):
                        val = (self.net_params.additional_params['off_ramps_pos'][1] - x)/(self.net_params.additional_params['off_ramps_pos'][1] - self.net_params.additional_params['off_ramps_pos'][0])
                        intention_reward += val
                        # print('2: ', cav_id, val)

        # penalty for frequent lane changing behavors
        drastic_lane_change_penalty = 0
        if self.drastic_veh_id:
            drastic_lane_change_penalty += len(self.drastic_veh_id) * unit


        # penalty for crashing
        total_crash_penalty = 0
        crash_ids = kwargs["fail"]
        total_crash_penalty = len(crash_ids) * unit
        # if crash_ids:
        #     print(crash_ids,total_crash_penalty)

        print(speed_reward,intention_reward,total_crash_penalty, drastic_lane_change_penalty)
        return  w_speed * speed_reward + \
                w_intention * intention_reward - \
                w_p_lane_change * total_crash_penalty - \
                w_p_crash * drastic_lane_change_penalty

    def apply_rl_actions(self, rl_actions=None):
        if isinstance(rl_actions,np.ndarray):
            # print(rl_actions.shape)
            # rl_actions = rl_actions.reshape((self.net_params.additional_params['num_cav'],3))
            rl_actions2 = rl_actions.copy()
            rl_actions2 -= 1
            rl_ids = self.observed_cavs
            drastic_veh = []
            for ind,veh_id in enumerate(rl_ids):
                if rl_actions2[ind]!=0 and (self.time_counter - self.k.vehicle.get_last_lc(veh_id)<50):
                    drastic_veh.append(veh_id)
                    # print("drastic lane change: ", veh_id)

            self.drastic_veh_id = drastic_veh
            self.k.vehicle.apply_lane_change(rl_ids, rl_actions2, 200)
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