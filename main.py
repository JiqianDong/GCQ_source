from flow.core.params import VehicleParams,InFlows
from flow.controllers import IDMController, RLController
from controller import SpecificMergeRouter,NearestMergeRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoLaneChangeParams

from network import HighwayRampsNetwork, ADDITIONAL_NET_PARAMS




#######################################################
########### Configurations
# TEST_SETTINGS = True
TEST_SETTINGS = False

RAY_RL = False


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


sim_params = SumoParams(sim_step=0.1, restart_instance=True, render=False)
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
# num_lanes = 3
# num_unique_intentions = len(set(intention_dic.values()))
# feature_size = 3 + num_lanes + num_unique_intentions
# rl_model = GraphicEncoder(feature_size)

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


def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None):
    """Return the relevant components of an RLlib experiment.
    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_cpus : int
        number of CPUs to run the experiment over
    n_rollouts : int
        number of rollouts per training iteration
    policy_graphs : dict, optional
        TODO
    policy_mapping_fn : function, optional
        TODO
    policies_to_train : list of str, optional
        TODO
    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    from flow.utils.registry import make_create_env
    from ray.tune.registry import register_env
    from copy import deepcopy
    import json
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class
    from flow.utils.rllib import FlowParamsEncoder, get_flow_params

    horizon = flow_params['env'].horizon

    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config["num_workers"] = n_cpus
    config["train_batch_size"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32, 32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = horizon

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    # if policy_graphs is not None:
    #     print("policy_graphs", policy_graphs)
    #     config['multiagent'].update({'policies': policy_graphs})
    # if policy_mapping_fn is not None:
    #     config['multiagent'].update(
    #         {'policy_mapping_fn': tune.function(policy_mapping_fn)})
    # if policies_to_train is not None:
    #     config['multiagent'].update({'policies_to_train': policies_to_train})

    create_env, gym_name = make_create_env(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


############ EXPERIMENTS ##############
if TEST_SETTINGS:
    from experiment import Experiment

    exp = Experiment(flow_params)

    # run the sumo simulation
    exp.run(1)
elif RAY_RL:
    import ray
    from ray.rllib.models import ModelCatalog
    from graph_model import GraphicPolicy
    from ray.tune import run_experiments


    n_cpus = 1
    # number of rollouts per training iteration
    n_rollouts = n_cpus * 4


    alg_run, gym_name, config = setup_exps_rllib(
        flow_params, n_cpus, n_rollouts)

    ModelCatalog.register_custom_model("graphic_policy", GraphicPolicy)

    config['model'] = {"custom_model":'graphic_policy'}

    ray.init(local_mode=True)
    exp_config = {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 20,
        "checkpoint_at_end": True,
        "max_failures": 999,
        "stop": {
            "training_iteration": 1,
        },
    }
    exp_config['restore'] = './'
    run_experiments({flow_params["exp_tag"]: exp_config})

else:
    from rl_experiments import Experiment

    exp = Experiment(flow_params)

    # run the sumo simulation
    exp.run(1)


