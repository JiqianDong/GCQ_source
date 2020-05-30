from spektral.layers import GraphConv
from tensorflow.keras.layers import Input, Dense, Lambda, Multiply, Reshape, Flatten
from tensorflow.keras.backend import gather, squeeze
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
from gym.spaces.box import Box
from gym.spaces import Discrete
from gym.spaces.dict import Dict


from agents.policy import *




class GraphicQNetworkKeras():
    def __init__(self, N,F, obs_space, action_space, num_outputs=3, model_config=None, name='graphic_policy_keras'):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.name = name
        self.base_model = self.build_model(N,F,num_outputs)

    def build_model(self,N,F,num_outputs):
        X_in = Input(shape=(N,F), name='X_in')
        A_in = Input(shape=(N,N), name='A_in')
        RL_indice = Input(shape=(N), name='rl_indice_in')

        ### Graphic convolution

        x = GraphConv(32, activation='relu',name='gcn1')([X_in, A_in])
        x = GraphConv(32, activation='relu',name='gcn2')([x, A_in])

        ### Policy network
        x1 = Dense(32,activation='relu',name='policy_1')(x)
        x2 = Dense(16,activation='relu',name='policy_2')(x1)

        ###  Action and filter
        x3 = Dense(num_outputs, activation='linear',name='policy_3')(x2)
        filt = Reshape((N,1),name='expend_dim')(RL_indice)
        qout = Multiply(name='filter')([x3,filt])

        model = Model(inputs = [X_in,A_in,RL_indice], outputs=[qout])
        print(model.summary())
        return model


# import pickle
# with open('training_data.pkl','rb') as f:
#     training_data = pickle.load(f)


from agents.memory import CustomerSequentialMemory

memory_buffer = CustomerSequentialMemory(limit=5000, window_length=1)
for obs,a, r, temin in zip(training_data['state'],training_data['action'], training_data['reward'], training_data['done']):
    if isinstance(a,np.ndarray):
        memory_buffer.append(obs,a,r,temin)


test_policy = greedy_q_policy()
start_policy = random_obs_policy()
train_policy = eps_greedy_q_policy()


from agents.processor import Jiqian_MultiInputProcessor
multi_input_processor = Jiqian_MultiInputProcessor(3)



model_config = {}
F = 9
N = 40

states = Box(low=-np.inf, high=np.inf, shape=(N,F), dtype=np.float32)
adjacency = Box(low=0, high=1, shape = (N,N), dtype=np.int32)
mask = Box(low=0, high=1, shape = (N,), dtype=np.int32)

obs_space = Dict({'states':states,'adjacency':adjacency,'mask':mask})
act_space = Box(low=0, high=1, shape = (N,), dtype=np.int32)
rl_model = GraphicQNetworkKeras(N,F, obs_space, act_space)

from agents.dqn import DQNAgent

my_dqn = DQNAgent(processor= multi_input_processor,
                  model = rl_model.base_model,
                  policy = train_policy,
                  test_policy=test_policy,
                  nb_total_agents = 40,
                  nb_actions = 3,
                  memory = memory_buffer,
                  nb_steps_warmup=10,
                  custom_model_objects={'GraphConv': GraphConv})


my_dqn.compile(Adam(0.001))