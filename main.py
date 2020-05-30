from flow.core.params import VehicleParams,InFlows
from flow.controllers import IDMController, RLController
from controller import SpecificMergeRouter,NearestMergeRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoLaneChangeParams

from network import HighwayRampsNetwork, ADDITIONAL_NET_PARAMS




#######################################################
########### Configurations
# TEST_SETTINGS = True
TEST_SETTINGS = False
TRAINING = True

RAY_RL = False

RENDER = True


NEAREST_MERGE = False
# NEAREST_MERGE = True

NUM_HUMAN = 20
NUM_MERGE_0 = 10
NUM_MERGE_1 = 10



VEH_COLORS = ['red','red'] if NEAREST_MERGE else ['red','green']


#######################################################



Router = NearestMergeRouter if NEAREST_MERGE else SpecificMergeRouter

vehicles = VehicleParams()
vehicles.add(veh_id="human",
             lane_change_params = SumoLaneChangeParams('strategic'),
             acceleration_controller=(IDMController, {}),
             routing_controller = (Router,{}),
             )

vehicles.add(veh_id="merge_0",
             lane_change_params = SumoLaneChangeParams('aggressive'),
             acceleration_controller=(RLController, {}),
             routing_controller = (Router,{}),
             color=VEH_COLORS[0])

vehicles.add(veh_id="merge_1",
             lane_change_params = SumoLaneChangeParams('aggressive'),
             acceleration_controller=(RLController, {}),
             routing_controller = (Router,{}),
             color=VEH_COLORS[1])

initial_config = InitialConfig(spacing='uniform')


inflow = InFlows()
inflow.add(veh_type="human",
           edge="highway_0",
           probability=0.2,
           depart_lane='random',
           route = 'highway_0',
           number = NUM_HUMAN)

inflow.add(veh_type="merge_0",
           edge="highway_0",
           probability = 0.1,
           depart_lane='random',
           route = 'highway_0',
           number = NUM_MERGE_0)

inflow.add(veh_type="merge_1",
           edge="highway_0",
           probability = 0.1,
           depart_lane='random',
           route = 'highway_0',
           number = NUM_MERGE_1)


sim_params = SumoParams(sim_step=0.1, restart_instance=True, render=RENDER)
# sim_params = SumoParams(sim_step=0.1, render=False)




from specific_environment import MergeEnv

intention_dic = {"human":0,"merge_0":1,"merge_1":1} if NEAREST_MERGE else {"human":0,"merge_0":1,"merge_1":2}

env_params = EnvParams(warmup_steps=50,additional_params={"intention":intention_dic})

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params['num_vehicles'] = NUM_HUMAN + NUM_MERGE_0 + NUM_MERGE_1
additional_net_params['num_cav'] = NUM_MERGE_0 + NUM_MERGE_1
additional_net_params['num_hv'] = NUM_HUMAN

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
# number of time steps
flow_params['env'].horizon = 8000


############ EXPERIMENTS ##############
if TEST_SETTINGS:
    from experiment import Experiment
    exp = Experiment(flow_params)

    # run the sumo simulation
    exp.run(1)
elif RAY_RL:
    from ray_rl_setting import *
else:
    from rl_experiments import Experiment

    exp = Experiment(flow_params)
    # run the sumo simulation
    exp.run(num_runs=1,training=TRAINING, num_human=NUM_HUMAN, num_cav=(NUM_MERGE_0+NUM_MERGE_1))


