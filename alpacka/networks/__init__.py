"""Network interface and its implementations."""

import gin
import gin.tf.external_configurables
import tensorflow as tf

from alpacka.networks import core
from alpacka.networks import keras
from alpacka.networks import tensorflow

gin.external_configurable(tf.nn.softmax_cross_entropy_with_logits,
                          module='tf.nn',
                          name='softmax_cross_entropy_with_logits')


def configure_network(network_class):
    return gin.external_configurable(
        network_class, module='alpacka.networks'
    )


Network = core.Network
TrainableNetwork = core.TrainableNetwork
DummyNetwork = configure_network(core.DummyNetwork)
KerasNetwork = configure_network(keras.KerasNetwork)
TFMetaGraphNetwork = configure_network(tensorflow.TFMetaGraphNetwork)
UnionNetwork = configure_network(core.UnionNetwork)
