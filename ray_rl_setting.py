import ray
from ray.rllib.models import ModelCatalog
from graph_model import GraphicPolicy
from ray.tune import run_experiments

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


