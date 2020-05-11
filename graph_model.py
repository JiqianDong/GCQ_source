from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from spektral.layers import GraphConv
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.backend import gather
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class GraphicEncoder():
    def __init__(self, F):
        self.model = self.build_model(F)

    def build_model(self,F):
        X_in = Input(shape=(F, ), name='X_in',batch_size=1)
        A_in = Input(shape=(None,), name='A_in',batch_size=1)
        RL_indice = Input(shape=(1,),name='RL_indice',dtype='int32',batch_size=1)

        x = GraphConv(32, activation='relu',name='gcn1')([X_in, A_in])
        x = GraphConv(32, activation='relu',name='gcn2')([x, A_in])
        x = Lambda(lambda x: gather(x,RL_indice),name='slice')(x)

        model = Model(inputs = [X_in,A_in,RL_indice], outputs=x)
        print(model.summary())
        return model
