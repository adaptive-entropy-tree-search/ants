"""Parameter schedules."""

import gin


@gin.configurable
class LinearAnnealing:

    def __init__(self, max_value, min_value, n_epochs):
        self._min_value = min_value
        self._slope = - (max_value - min_value) / n_epochs
        self._intersect = max_value

    def __call__(self, epoch):
        return max(self._min_value, self._slope * epoch + self._intersect)


@gin.configurable
class RsqrtAnnealing:

    def __init__(self, max_value=1, scale=1):
        self._max_value = max_value
        self._scale = scale

    def __call__(self, epoch):
        return self._max_value / (1 + epoch / self._scale) ** 0.5
