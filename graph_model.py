from spektral.layers import GraphConv
from tensorflow.keras.layers import Input, Dense, Lambda, Multiply, Reshape, Flatten, Masking, LSTM, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2




class GraphicQNetworkKeras():
    def __init__(self, N,F, obs_space, action_space, num_outputs=3, model_config=None, name='graphic_policy_keras'):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.name = name
        self.base_model = self.build_model(N,F,num_outputs)
        print("new model")

    def build_model(self,N,F,num_outputs):
        X_in = Input(shape=(N,F), name='X_in')
        A_in = Input(shape=(N,N), name='A_in')
        RL_indice = Input(shape=(N), name='rl_indice_in')

        ### Encoder
        x = Dense(32,activation='relu',name='encoder_1')(X_in)
        x = Dense(32,activation='relu',name='encoder_2')(x)

        ### Graphic convolution

        x = GraphConv(32, activation='relu',name='gcn1')([x, A_in])
        # x = GraphConv(32, activation='relu',name='gcn2')([x, A_in])

        ### Policy network
        x1 = Dense(32,activation='relu',name='policy_1')(x)
        x1 = GraphConv(32, activation='relu',name='gcn2')([x1, A_in])
        x1 = Dense(32,activation='relu',name='policy_add')(x1)
        x2 = Dense(16,activation='relu',name='policy_2')(x1)

        ###  Action and filter
        x3 = Dense( num_outputs, activation='linear',name='policy_3')(x2)
        filt = Reshape((N,1),name='expend_dim')(RL_indice)
        qout = Multiply(name='filter')([x3,filt])

        model = Model(inputs = [X_in,A_in,RL_indice], outputs=[qout])
        # print(model.summary())
        return model


class LstmQNetworkKeras():
    def __init__(self, N,F, obs_space, action_space, num_outputs=3, model_config=None, name='graphic_policy_keras'):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.name = name
        self.base_model = self.build_model(N,F,num_outputs)

    def build_model(self,N,F,num_outputs):

        X_in = Input(shape=(N,F), name='X_in')
        A_in = Input(shape=(N,N), name='A_in')
        x = Dense(32,activation='relu',name='encoder_1')(X_in)
        x = Dense(32,activation='relu',name='encoder_2')(x)

        RL_indice = Input(shape=(N), name='rl_indice_in')


        x = Masking(mask_value=0,input_shape=(N,32))(x)

        ### LSTM fusion
        x = LSTM(32,return_sequences=True)(x)

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

class GraphicQNetworkKeras2():
    def __init__(self, N,F, obs_space, action_space, num_outputs=3, model_config=None, name='graphic_q_keras'):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.name = name
        self.base_model = self.build_model(N,F,num_outputs)

    def build_model(self,N,F,num_outputs):

        X_in = Input(shape=(N,F), name='X_in')
        A_in = Input(shape=(N,N), name='A_in')
        RL_indice = Input(shape=(N), name='rl_indice_in')

        ### Encoder
        x = Dense(32,activation='relu',name='encoder_1')(X_in)
        x = Dense(32,activation='relu',name='encoder_2')(x)


        ### Graphic convolution

        x1 = GraphConv(32, activation='relu',name='gcn1')([x, A_in])
        x1 = Dense(32,activation='relu',name='post_gcn_1')(x1)


        # x2 = GraphConv(32, activation='relu',name='gcn2')([x1, A_in])
        # x2 = Dense(32,activation='relu',name='post_gcn_2')(x2)


        ###  Action and filter
        x3 = Concatenate()([x,x1])
        x3 = Dense(64, activation='relu',name='policy_1')(x3)
        x3 = Dense(32, activation='relu',name='policy_2')(x3)

        x3 = Dense(num_outputs, activation='linear',name='policy_output')(x3)
        mask = Reshape((N,1),name='expend_dim')(RL_indice)
        qout = Multiply(name='filter')([x3,mask])

        model = Model(inputs = [X_in,A_in,RL_indice], outputs=[qout])
        print(model.summary())
        return model


