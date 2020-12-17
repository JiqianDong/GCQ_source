"""Contains an experiment class for running simulations."""
from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
import datetime
import logging
import time
import os
import numpy as np
import json

class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a network and environment. In order to use
    it to run an network and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> flow_params = dict(...)  # see the examples in exp_config
        >>> exp = Experiment(flow_params)  # for some experiment configuration
        >>> exp.run(num_runs=1)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> flow_params['sim'] = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    custom_callables : dict < str, lambda >
        strings and lambda functions corresponding to some information we want
        to extract from the environment. The lambda will be called at each step
        to extract information from the env and it will be stored in a dict
        keyed by the str.
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self, flow_params, custom_callables=None):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
            flow-specific parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")


    def run(self,num_runs,training,num_human,actual_num_human,num_cav, model, debug):
        model_name = model+'_hv_'+str(num_human)+'_cav_'+str(num_cav)

        if debug:
            nb_steps_warmup = 30
            batch_size = 10
            total_steps = 200
            log_interval = 40
            nb_max_episode_steps = 20
            gamma = 0.99
        else:
            nb_steps_warmup = 200000
            batch_size = 32
            total_steps = 800000
            log_interval = 4000
            nb_max_episode_steps = 2500
            gamma = 0.99



        F = 2 + self.env.net_params.additional_params['highway_lanes'] + self.env.n_unique_intentions # input feature size
        N = num_human + num_cav
        A = 3

        from gym.spaces.box import Box
        from gym.spaces import Discrete
        from gym.spaces.dict import Dict
        states = Box(low=-np.inf, high=np.inf, shape=(N,F), dtype=np.float32)
        adjacency = Box(low=0, high=1, shape = (N,N), dtype=np.int32)
        mask = Box(low=0, high=1, shape = (N,), dtype=np.int32)

        obs_space = Dict({'states':states,'adjacency':adjacency,'mask':mask})
        act_space = Box(low=0, high=1, shape = (N,), dtype=np.int32)

        from graph_model import GraphicQNetworkKeras, LstmQNetworkKeras, GraphicQNetworkKeras2
        from agents.memory import CustomerSequentialMemory
        from agents.processor import Jiqian_MultiInputProcessor
        from agents.dqn import DQNAgent
        from agents.policy import eps_greedy_q_policy,greedy_q_policy,random_obs_policy
        from spektral.layers import GraphConv
        from tensorflow.keras.optimizers import Adam
        import tensorflow as tf

        memory_buffer = CustomerSequentialMemory(limit=100000, window_length=1)
        multi_input_processor = Jiqian_MultiInputProcessor(A)


        if model=='gcn':
            rl_model = GraphicQNetworkKeras2(N,F,obs_space,act_space)
        elif model == 'lstm':
            rl_model = LstmQNetworkKeras(N,F,obs_space,act_space)
        else:
            raise NotImplementedError


        my_dqn = DQNAgent(processor= multi_input_processor,
                          model = rl_model.base_model,
                          policy = eps_greedy_q_policy(0.3),
                          test_policy = greedy_q_policy(),
                          start_policy = random_obs_policy(),
                          nb_total_agents = N,
                          nb_actions = A,
                          memory = memory_buffer,
                          nb_steps_warmup=nb_steps_warmup,
                          batch_size=batch_size,
                          gamma = gamma,
                          custom_model_objects={'GraphConv': GraphConv})

        my_dqn.compile(Adam(0.001))

        if training:

            logdir = "./logs/"
            history_file = "./logs/" + model_name + '_training_hist.txt'
            try:
                # os.rmdir(logdir)
                os.remove(history_file)

            except:
                pass


            from agents.rl_lib.callbacks import FileLogger

            # from tensorflow.python.keras.callbacks import TensorBoard
            # tensorboard_callback = TensorBoard(log_dir=logdir,histogram_freq=1,write_graph=True,update_freq='batch')

            file_log = FileLogger(history_file)
            history = my_dqn.fit(self.env, nb_steps=total_steps, nb_max_episode_steps=nb_max_episode_steps,visualize=False, verbose=1, log_interval=log_interval, callbacks=[file_log])
            my_dqn.save_weights('./models/dqn_{}.h5f'.format(model_name), overwrite=True)

            from generate_training_plots import plot_training


            plot_training(logdir)

        else:
            history_file = "./logs/test/{}_cav_{}_hv_{}_testing_hist.txt".format(model,num_cav,actual_num_human)

            my_dqn.load_weights('./models/dqn_{}.h5f'.format(model_name))
            print("succssfully loaded")
            hist = my_dqn.test(self.env,nb_episodes=num_runs)
            # print(hist.history)

            with open(history_file,'w') as f:
                json.dump(hist.history, f)


