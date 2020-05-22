from spektral.layers import GraphConv
from tensorflow.keras.layers import Input, Dense, Lambda, Multiply, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

class GraphicPolicy(TFModelV2):
    def __init__(self, N,F, obs_space, action_space, num_outputs=3, model_config=None, name='graphic_policy_rllib'):
        super(GraphicPolicy,self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.base_model = self.build_model(N,F,num_outputs)
        self.register_variables(self.base_model.variables)

    def build_model(self,N,F,num_outputs):
        X_in = Input(shape=(N,F), name='X_in')
        A_in = Input(shape=(N,N), name='A_in')
        RL_indice = Input(shape=(N), name='mask')

        ### Graphic convolution

        x = GraphConv(32, activation='relu',name='gcn1')([X_in, A_in])
        x = GraphConv(32, activation='relu',name='gcn2')([x, A_in])


        ### Policy network

        x1 = Dense(32,activation='relu',name='policy_1')(x)
        x2 = Dense(16,activation='relu',name='policy_2')(x1)


        ###  Action and filter
        x3 = Dense(num_outputs, activation='relu',name='policy_3')(x2)
        mask = Reshape((N,1),name='expend_dim')(RL_indice)
        out = Multiply(name='filter')([x3,mask])


        #### Value out
        x2 = Flatten(name='flatten')(x2)
        value = Dense(1,activation='None',name='value_out')(x2)


        model = Model(inputs = [X_in,A_in,RL_indice], outputs=[out,value])
        print(model.summary())
        return model

    def forward(self, input_dict, state, seq_lens):
        obs = input_dic['obs']
        model_out, self._value_out = self.base_model({X_in:obs['states'],A_in:obs['adjacency'],RL_indice:obs['mask']})
        return model_out,state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])



class GraphicPolicyKeras():
    def __init__(self, N,F, obs_space, action_space, num_outputs=3, model_config=None, name='graphic_policy_keras'):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.name = name
        self.base_model = self.build_model(N,F,num_outputs)

    def build_model(self,N,F,num_outputs):
        X_in = Input(shape=(N,F), name='X_in')
        A_in = Input(shape=(N,N), name='A_in')
        RL_indice = Input(shape=(N), name='mask')

        ### Graphic convolution

        x = GraphConv(32, activation='relu',name='gcn1')([X_in, A_in])
        x = GraphConv(32, activation='relu',name='gcn2')([x, A_in])

        ### Policy network
        x1 = Dense(32,activation='relu',name='policy_1')(x)
        x2 = Dense(16,activation='relu',name='policy_2')(x1)

        ###  Action and filter
        x3 = Dense(num_outputs, activation='relu',name='policy_3')(x2)
        mask = Reshape((N,1),name='expend_dim')(RL_indice)
        out = Multiply(name='filter')([x3,mask])
        out = Flatten(name='action_flatten')(out)

        #### Value out
        x2 = Flatten(name='value_flatten')(x2)
        value = Dense(1,activation='None',name='value_out')(x2)

        model = Model(inputs = [X_in,A_in,RL_indice], outputs=[out,value])
        print(model.summary())
        return model

    def forward(self, input_dict, state, seq_lens):
        obs = input_dic['obs']
        model_out, self._value_out = self.base_model({X_in:obs['states'],A_in:obs['adjacency'],RL_indice:obs['mask']})
        return model_out,state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class GraphicQNetworkKeras():
    def __init__(self, N,F, obs_space, action_space, num_outputs=3, model_config=None, name='graphic_policy_keras'):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.name = name
        self.base_model = self.build_model(N,F,self.num_outputs)

    def build_model(self,N,F,num_outputs):
        X_in = Input(shape=(N,F), name='X_in')
        A_in = Input(shape=(N,N), name='A_in')
        RL_indice = Input(shape=(N), name='mask')

        ### Graphic convolution

        x = GraphConv(32, activation='relu',name='gcn1')([X_in, A_in])
        x = GraphConv(32, activation='relu',name='gcn2')([x, A_in])

        ### Policy network
        x1 = Dense(32,activation='relu',name='policy_1')(x)
        x2 = Dense(16,activation='relu',name='policy_2')(x1)

        ###  Action and filter
        x3 = Dense(num_outputs, activation='linear',name='policy_3')(x2)
        mask = Reshape((N,1),name='expend_dim')(RL_indice)
        qout = Multiply(name='filter')([x3,mask])

        model = Model(inputs = [X_in,A_in,RL_indice], outputs=[qout])
        print(model.summary())
        return model



