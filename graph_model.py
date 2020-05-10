import tensorflow as tf

from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from spektral.layers import GraphConv, MinCutPool, GlobalSumPool



class GraphicEncoder():
    def __init__(self,):
        self.model = self.build_model()

    def build_model(self):
        pass
