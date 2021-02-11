"""Network interface implementation using the Keras framework."""

import functools

import gin
import numpy as np
import tensorflow as tf
from tensorflow import keras

from alpacka import data
from alpacka.networks import core


def _is_pointwise_loss(name):
    return name.startswith('mean_') or name in {
        'binary_crossentropy', 'hinge', 'squared_hinge', 'poisson', 'log_cosh',
        'huber_loss', 'cosine_similarity',
        'mse', 'mae', 'mape', 'msle', 'bce', 'logcosh', 'huber'
    }


def pointwise_loss(loss_fn):
    def new_loss_fn(y_true, y_pred):
        assert y_true.shape.is_compatible_with(y_pred.shape)
        loss = loss_fn(y_true[..., None], y_pred[..., None])
        assert loss.shape.is_compatible_with(y_true.shape)
        return loss

    new_loss_fn.__name__ = loss_fn.__name__ + '_pointwise'
    return new_loss_fn


def interdependent_loss(loss_fn):
    def new_loss_fn(y_true, y_pred):
        assert y_true.shape.is_compatible_with(y_pred.shape)
        loss = loss_fn(y_true, y_pred)
        return tf.broadcast_to(loss[..., None], shape=tf.shape(y_pred))

    new_loss_fn.__name__ = loss_fn.__name__ + '_interdependent'
    return new_loss_fn


def _wrap_loss(loss_or_name):
    if isinstance(loss_or_name, str):

        name = loss_or_name
        loss = keras.losses.get(name)
    elif 'tensorflow.python.keras' in loss_or_name.__module__:

        loss = loss_or_name
        name = loss.name
    else:

        return loss_or_name

    if _is_pointwise_loss(name):
        return pointwise_loss(loss)
    else:
        return interdependent_loss(loss)


class AddMask(keras.layers.Layer):
    supports_masking = True

    def compute_mask(self, inputs, mask=None):
        assert mask is None
        (_, mask) = inputs
        return mask

    def call(self, inputs, **kwargs):
        del kwargs
        (true_input, _) = inputs
        return true_input

    def compute_output_shape(self, input_shape):
        (input_shape, _) = input_shape
        return input_shape


def _make_inputs(input_signature):
    def init_layer(signature):
        return keras.Input(shape=signature.shape, dtype=signature.dtype)

    return data.nested_map(init_layer, input_signature)


def _make_output_heads(hidden, output_signature, output_activation, zero_init):
    masks = _make_inputs(output_signature)

    def init_head(layer, signature, activation, mask):
        assert signature.dtype == np.float32
        depth = signature.shape[-1]
        kwargs = {'activation': activation}
        if zero_init:
            kwargs['kernel_initializer'] = 'zeros'
            kwargs['bias_initializer'] = 'zeros'
        head = keras.layers.Dense(depth, **kwargs)(layer)
        return AddMask()((head, mask))

    if tf.is_tensor(hidden):
        hidden = data.nested_map(lambda _: hidden, output_signature)

    heads = data.nested_zip_with(
        init_head, (hidden, output_signature, output_activation, masks)
    )
    return (heads, masks)


@gin.configurable
def mlp(
        network_signature,
        hidden_sizes=(32,),
        activation='relu',
        output_activation=None,
        output_zero_init=False,
):
    inputs = _make_inputs(network_signature.input)

    x = inputs
    for h in hidden_sizes:
        x = keras.layers.Dense(h, activation=activation)(x)

    (outputs, masks) = _make_output_heads(
        x, network_signature.output, output_activation, output_zero_init
    )
    return keras.Model(inputs=(inputs, masks), outputs=outputs)


@gin.configurable
def additive_injector(primary, auxiliary):
    primary = keras.layers.LayerNormalization()(primary)
    auxiliary = keras.layers.LayerNormalization(center=False)(auxiliary)
    return keras.layers.Add()((primary, auxiliary))


def _inject_auxiliary_input(primary, auxiliary, injector):
    if injector is not None:
        depth = primary.shape[-1]
        auxiliary = keras.layers.Dense(depth)(auxiliary)
        primary = injector(primary, auxiliary)
    return primary


@gin.configurable
def convnet_mnist(
        network_signature,
        n_conv_layers=5,
        d_conv=64,
        d_ff=128,
        activation='relu',
        aux_input_injector=None,
        output_activation=None,
        output_zero_init=False,
        global_average_pooling=False,
        strides=(1, 1),
):
    inputs = _make_inputs(network_signature.input)

    if aux_input_injector is None:
        x = inputs
        aux_input = None
    else:
        (x, aux_input) = inputs

    for _ in range(n_conv_layers):
        x = keras.layers.Conv2D(
            d_conv,
            kernel_size=(3, 3),
            padding='same',
            activation=activation,
            strides=strides,
        )(x)
    if global_average_pooling:
        x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(d_ff)(x)
    x = _inject_auxiliary_input(x, aux_input, aux_input_injector)
    x = keras.layers.Activation(activation)(x)

    (outputs, masks) = _make_output_heads(
        x, network_signature.output, output_activation, output_zero_init
    )
    return keras.Model(inputs=(inputs, masks), outputs=outputs)


@gin.configurable
def convnet_dqn(
        network_signature,
        d_conv=64,
        d_ff=512,
        aux_input_injector=None,
        output_activation=None,
        output_zero_init=False,
):
    inputs = _make_inputs(network_signature.input)

    if aux_input_injector is None:
        x = inputs
        aux_input = None
    else:
        (x, aux_input) = inputs

    for (kernel, stride) in [(8, 4), (4, 2), (3, 1)]:
        x = keras.layers.Conv2D(
            d_conv,
            kernel_size=(kernel, kernel),
            strides=(stride, stride),
            padding='same',
            activation='relu',
        )(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(d_ff)(x)
    x = _inject_auxiliary_input(x, aux_input, aux_input_injector)
    x = keras.layers.Activation('relu')(x)

    (outputs, masks) = _make_output_heads(
        x, network_signature.output, output_activation, output_zero_init
    )
    return keras.Model(inputs=(inputs, masks), outputs=outputs)


def _spread_action_over_the_board(obs, action):
    assert len(obs.shape) == 4
    assert len(action.shape) == 2

    n_actions = action.shape[-1]
    action_shape = tf.constant((-1, 1, 1, n_actions), dtype=tf.int32)
    action = tf.reshape(action, action_shape)

    multipliers = [1, obs.shape[1], obs.shape[2], 1]
    action = tf.tile(action, tf.constant(multipliers, dtype=tf.int32))

    return tf.concat([obs, action], axis=-1)


@gin.configurable
def fcn_for_env_model(
        network_signature,
        cnn_channels=64,
        cnn_n_layers=2,
        cnn_kernel_size=(5, 5),
        cnn_strides=(1, 1),
        cnn_final_pool_size=(1, 1),
        output_activation=None,
        batch_norm=False,
        output_zero_init=False,
):
    inputs = _make_inputs(network_signature.input)

    observation = inputs['observation']
    action = inputs['action']

    x = _spread_action_over_the_board(observation, action)

    for _ in range(cnn_n_layers):
        x = keras.layers.Conv2D(
            cnn_channels, kernel_size=cnn_kernel_size, strides=cnn_strides,
            padding='same', activation='relu'
        )(x)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling2D(pool_size=cnn_final_pool_size)(x)

    avg_channels = keras.layers.GlobalAveragePooling2D()(x)

    if output_activation is None:
        output_activation = {
            'next_observation': keras.activations.softmax,
            'reward': None,
            'done': keras.activations.sigmoid
        }

    final_layers = {
        'next_observation': x,
        'reward': avg_channels,
        'done': avg_channels,
    }

    (outputs, masks) = _make_output_heads(
        final_layers, network_signature.output, output_activation,
        output_zero_init
    )
    return keras.Model(inputs=(inputs, masks), outputs=outputs)


class KerasNetwork(core.TrainableNetwork):

    def __init__(
            self,
            network_signature,
            model_fn=mlp,
            optimizer='adam',
            loss='mean_squared_error',
            loss_weights=None,
            weight_decay=0.0,
            metrics=None,
            train_callbacks=None,
            seed=None,
            **compile_kwargs
    ):
        super().__init__(network_signature)
        self._network_signature = network_signature
        self._model = model_fn(network_signature)
        self._add_weight_decay(self._model, weight_decay)

        if seed is not None:
            tf.random.set_seed(seed)

        metrics = metrics or []
        (loss, metrics) = data.nested_map(_wrap_loss, (loss, metrics))
        self._model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            **compile_kwargs
        )

        self.train_callbacks = train_callbacks or []

    @staticmethod
    def _add_weight_decay(model, weight_decay):
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                layer.add_loss(functools.partial(
                    keras.regularizers.l2(weight_decay), layer.kernel
                ))

    def train(self, data_stream, n_steps):
        def masked_data_stream():
            for (inp, target, mask) in data_stream():
                yield ((inp, mask), target)

        def dtypes(tensors):
            return data.nested_map(lambda x: x.dtype, tensors)

        def shapes(tensors):
            return data.nested_map(lambda x: x.shape, tensors)

        dataset = tf.data.Dataset.from_generator(
            generator=masked_data_stream,
            output_types=dtypes((self._model.input, self._model.output)),
            output_shapes=shapes((self._model.input, self._model.output)),
        )

        history = self._model.fit(
            dataset, epochs=1, verbose=0, steps_per_epoch=n_steps,
            callbacks=self.train_callbacks
        )

        return {name: values[0] for (name, values) in history.history.items()}

    def predict(self, inputs):
        some_leaf_shape = data.choose_leaf(inputs).shape
        assert some_leaf_shape, 'KerasNetwork only works on batched inputs.'
        batch_size = some_leaf_shape[0]

        def one_array(signature):
            return np.ones(
                shape=((batch_size,) + signature.shape), dtype=signature.dtype
            )

        masks = data.nested_map(one_array, self._network_signature.output)

        return self._model.predict_on_batch((inputs, masks))

    @property
    def params(self):
        return self._model.get_weights()

    @params.setter
    def params(self, new_params):
        self._model.set_weights(new_params)

    def save(self, checkpoint_path):
        self._model.save_weights(checkpoint_path, save_format='h5')

    def restore(self, checkpoint_path):
        self._model.load_weights(checkpoint_path)
