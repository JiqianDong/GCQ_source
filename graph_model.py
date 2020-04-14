import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow.keras.layers import Dense


flags = tf.app.flags
FLAGS = flags.FLAGS




class GraphConvolution(tf.keras.layers.Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__()
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
    def build():


    def call(self, inputs, adjacency):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(adjacency, x)
        outputs = self.act(x)
        return outputs



class GCNModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, placeholders):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.adj = placeholders['adj']
        # self.dropout = placeholders['dropout']
        self.dropout = 0
        self.build()



    def _build(self):
        self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        adj=self.adj,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.hidden3 = Dense()(self.embeddings)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}



class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)





        # self.z_mean = self.embeddings
        # self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
        #                               act=lambda x: x,
        #                               logging=self.logging)(self.embeddings)