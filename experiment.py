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

    def run(self, num_runs, num_cav=20, num_human = 20,  num_merge_0=None, num_merge_1=None, rl_actions=None, convert_to_csv=False):
        """Run the given network for a set number of runs.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file

        Returns
        -------
        info_dict : dict < str, Any >
            contains returns, average speed per step
        """
        num_steps = 10000 #maximum steps

        if rl_actions is None:
            def rl_actions(*_):
                rl_ids = self.env.k.vehicle.get_rl_ids()
                if rl_ids:
                    if (self.env.time_counter-self.env.env_params.warmup_steps)%100 == 99:
                    # if (self.env.time_counter-self.env.env_params.warmup_steps)%30 == 0:
                        # print('lane change')
                        return np.random.choice(3,len(rl_ids))
                        # return np.zeros(len(rl_ids))
                        # return np.ones(len(rl_ids))
                    else:
                        return np.ones(len(rl_ids))
                else:
                    return None
   
        rewards = []
        total_steps = []

        for i in range(num_runs):
            ret = 0
            state = self.env.reset()

            for j in range(num_steps):
                t0 = time.time()
                action = rl_actions(state)
                state, reward, done, _ = self.env.step(action)

                ret += reward

                if done:
                    print('finished with step: ',j)
                    break
            rewards.append(ret)
            total_steps.append(j+1)

        if num_merge_0 is not None:
            file_name = "./logs/test/vary_ramp_popularity/{}_cav0_{}_cav1_{}_hv_{}_testing_hist.txt".format("rule_based",num_merge_0,num_merge_1,num_human)
        else:
            file_name = "./logs/test/{}_cav_{}_hv_{}_testing_hist2.txt".format("rule_based",num_cav,num_human)
        with open(file_name,'w') as f:
            json.dump({'episode_reward':rewards,'num_steps':total_steps},f)
        self.env.terminate()

