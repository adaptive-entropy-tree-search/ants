"""Deep learning framework-agnostic interface for neural networks."""

import os

import gin

from alpacka import data
from alpacka.utils import transformations


class Network:

    def __init__(self, network_signature):
        self._network_signature = network_signature

    def clone(self):
        new_network = type(self)(network_signature=self._network_signature)
        new_network.params = self.params
        return new_network

    def predict(self, inputs):
        raise NotImplementedError

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, new_params):
        raise NotImplementedError

    def save(self, checkpoint_path):
        raise NotImplementedError

    def restore(self, checkpoint_path):
        raise NotImplementedError


class TrainableNetwork(Network):

    def train(self, data_stream, n_steps):
        raise NotImplementedError


class DummyNetwork(TrainableNetwork):

    def __init__(self, network_signature):
        super().__init__(network_signature)
        self._params = ''

    def train(self, data_stream, n_steps):
        del data_stream
        return {}

    def predict(self, inputs):
        if self._network_signature is None:

            return inputs
        else:

            input_shapes = data.nested_map(lambda x: x.shape[1:], inputs)
            expected_shapes = data.nested_map(
                lambda x: x.shape, self._network_signature.input
            )
            assert input_shapes == expected_shapes, (
                f'Incorrect input shapes: {input_shapes} != {expected_shapes}.'
            )

            batch_size = data.choose_leaf(inputs).shape[0]
            return data.zero_pytree(
                self._network_signature.output, shape_prefix=(batch_size,)
            )

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params

    def save(self, checkpoint_path):
        with open(checkpoint_path, 'w') as f:
            f.write(self._params)

    def restore(self, checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            self._params = f.read()


class UnionNetwork(Network):

    def __init__(self, network_signature, request_to_network=gin.REQUIRED):
        super().__init__(network_signature)

        [networks, signatures] = [
            transformations.map_dict_keys(dictionary, data.request_type_id)
            for dictionary in [request_to_network, network_signature]
        ]

        self._networks = {
            type_id: network_fn(sig)
            for type_id, (network_fn, sig) in transformations.zip_dicts_strict(
                networks, signatures
            ).items()
        }

    def predict(self, inputs):
        type_id = data.request_type_id(type(inputs))
        network = self._networks[type_id]
        return network.predict(inputs.value)

    @property
    def subnetworks(self):
        return self._networks

    @property
    def params(self):
        return {
            type_id: network.params
            for type_id, network in self._networks.items()
        }

    @params.setter
    def params(self, new_params):
        if new_params.keys() != self._networks.keys():
            raise ValueError(
                'Keys of new_params do not match stored networks.\n'
                f'{new_params.keys()} != {self._networks.keys()}'
            )

        for type_id, network in self._networks.items():
            network.params = new_params[type_id]

    def save(self, checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
        for (slug, network) in self.subnetworks.items():
            network.save(os.path.join(checkpoint_path, slug))

    def restore(self, checkpoint_path):
        for (slug, network) in self.subnetworks.items():
            network.restore(os.path.join(checkpoint_path, slug))
