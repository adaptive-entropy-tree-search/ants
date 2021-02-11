"""Base class for trainers."""


class Trainer:

    def __init__(self, network_signature):
        del network_signature

    def add_episode(self, episode):
        raise NotImplementedError

    def train_epoch(self, network):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def restore(self, path):
        raise NotImplementedError
