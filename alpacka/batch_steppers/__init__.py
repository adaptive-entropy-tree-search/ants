"""Batch steppers."""

import gin

from alpacka.batch_steppers import local
from alpacka.batch_steppers import process


def configure_batch_stapper(batch_stepper_class):
    return gin.external_configurable(
        batch_stepper_class, module='alpacka.batch_steppers'
    )


LocalBatchStepper = configure_batch_stapper(local.LocalBatchStepper)
ProcessBatchStepper = configure_batch_stapper(process.ProcessBatchStepper)
