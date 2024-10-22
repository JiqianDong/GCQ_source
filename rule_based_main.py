from flow.core.params import VehicleParams,InFlows,SumoCarFollowingParams,SumoParams, EnvParams, InitialConfig, NetParams, SumoLaneChangeParams
from flow.controllers import IDMController, RLController
from controller import SpecificMergeRouter,NearestMergeRouter
# from flow.core.params import

from network import HighwayRampsNetwork, ADDITIONAL_NET_PARAMS




#######################################################
########### Configurations
# actual_num_human_list = [10,20,30,40,50]

actual_num_human = 20
actual_num_cav_list = [(0,20),(5,15),(15,5),(20,0)]

for (NUM_MERGE_0,NUM_MERGE_1) in actual_num_cav_list:

# for actual_num_human in actual_num_human_list:
    TEST_SETTINGS = True

    RENDER = False
    # RENDER = True


    NEAREST_MERGE = False
    # NEAREST_MERGE = True




    NUM_HUMAN = 20


    # NUM_MERGE_0 = 10
    # NUM_MERGE_1 = 10

    MAX_CAV_SPEED = 14
    MAX_HV_SPEED = 10


    VEH_COLORS = ['red','red'] if NEAREST_MERGE else ['red','green']


    #######################################################


    # Router = NearestMergeRouter if NEAREST_MERGE else SpecificMergeRouter

    vehicles = VehicleParams()
    vehicles.add(veh_id="human",
                 lane_change_params = SumoLaneChangeParams('only_strategic_safe'),
                 car_following_params = SumoCarFollowingParams(speed_mode='right_of_way',min_gap=5, tau=0.5, max_speed=MAX_HV_SPEED),
                 acceleration_controller=(IDMController, {}),
                 )

    vehicles.add(veh_id="merge_0",
                 lane_change_params = SumoLaneChangeParams('only_strategic_safe'),
                 car_following_params = SumoCarFollowingParams(speed_mode='no_collide',min_gap=1, tau=0.5, max_speed=MAX_CAV_SPEED),
                 acceleration_controller=(RLController, {}),
                 color=VEH_COLORS[0])

    vehicles.add(veh_id="merge_1",
                 lane_change_params = SumoLaneChangeParams('only_strategic_safe'),
                 car_following_params = SumoCarFollowingParams(speed_mode='no_collide',min_gap=1, tau=0.5, max_speed=MAX_CAV_SPEED),
                 acceleration_controller=(RLController, {}),
                 color=VEH_COLORS[1])

    initial_config = InitialConfig(spacing='uniform')


    inflow = InFlows()
    inflow.add(veh_type="human",
               edge="highway_0",
               probability=actual_num_human/100,
               depart_lane='random',
               depart_speed = 'random',
               route = 'highway_0',
               number = actual_num_human)

    inflow.add(veh_type="merge_0",
               edge="highway_0",
               probability = 0.1,
               depart_lane='random',
               depart_speed = 'random',
               route = 'merge_0',
               number = NUM_MERGE_0)

    inflow.add(veh_type="merge_1",
               edge="highway_0",
               probability = 0.1,
               depart_lane='random',
               depart_speed = 'random',
               route = 'merge_1',
               number = NUM_MERGE_1)


    sim_params = SumoParams(sim_step=0.1, restart_instance=True, render=RENDER,seed=None)


    from specific_environment import MergeEnv

    intention_dic = {"human":0,"merge_0":1,"merge_1":1} if NEAREST_MERGE else {"human":0,"merge_0":1,"merge_1":2}
    terminal_edges = ['off_ramp_0','off_ramp_1','highway_2']

    env_params = EnvParams(warmup_steps=50,
                           additional_params={"intention":intention_dic,
                                              "max_cav_speed":MAX_CAV_SPEED,
                                              "max_hv_speed":MAX_HV_SPEED})

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params['num_vehicles'] = NUM_HUMAN + NUM_MERGE_0 + NUM_MERGE_1
    additional_net_params['num_cav'] = NUM_MERGE_0 + NUM_MERGE_1
    additional_net_params['num_hv'] = NUM_HUMAN
    additional_net_params['terminal_edges'] = terminal_edges

    net_params = NetParams(inflows=inflow, additional_params=additional_net_params)

    network = HighwayRampsNetwork("highway_ramp",vehicles,net_params,initial_config)


    ############ BUILD RL MODEL ##############

    flow_params = dict(
        exp_tag='test_network',
        env_name=MergeEnv,
        network=network,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config
    )
    # # number of time steps
    flow_params['env'].horizon = 2000


    ############ EXPERIMENTS ##############
    if TEST_SETTINGS:
        print("this is the run for the baseline model")
        from experiment import Experiment
        exp = Experiment(flow_params)

        # run the sumo simulation
        # exp.run(10,num_cav=(NUM_MERGE_0+NUM_MERGE_1),num_human=actual_num_human)  
        exp.run(10,num_cav=(NUM_MERGE_0+NUM_MERGE_1),num_merge_0=NUM_MERGE_0, num_merge_1=NUM_MERGE_1, num_human=actual_num_human) # For varying the popularity
        



