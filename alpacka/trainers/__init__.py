"""Neural network trainers."""

import gin

from alpacka.trainers import dummy
from alpacka.trainers import supervised
from alpacka.trainers.base import *


def configure_trainer(trainer_class):
    return gin.external_configurable(
        trainer_class, module='alpacka.trainers'
    )


DummyTrainer = configure_trainer(dummy.DummyTrainer)
SupervisedTrainer = configure_trainer(supervised.SupervisedTrainer)
