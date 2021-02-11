"""Math functions."""

import gin
import numpy as np


def log_sum_exp(logits, keep_last_dim=False):
    logits = np.array(logits)
    baseline = np.max(logits, axis=-1, keepdims=True)
    result = np.log(
        np.sum(np.exp(logits - baseline), axis=-1, keepdims=True)
    ) + baseline
    if not keep_last_dim:
        result = np.squeeze(result, axis=-1)
    return result


def log_mean_exp(logits, keep_last_dim=False):
    logits = np.array(logits)
    return log_sum_exp(logits, keep_last_dim) - np.log(logits.shape[-1])


def log_softmax(logits):
    return logits - log_sum_exp(logits, keep_last_dim=True)


def softmax(logits):
    return np.exp(log_softmax(logits))


def _validate_categorical_params(logits, probs):
    if (logits is None) == (probs is None):
        raise ValueError(
            'Either logits or probs must be provided (exactly one has to be '
            'not None).'
        )

    if probs is not None:
        if np.any(probs < 0):
            raise ValueError('Some probabilities are negative.')

        if not np.allclose(np.sum(probs, axis=-1), 1):
            raise ValueError('Probabilities don\'t sum to one.')


def categorical_entropy(logits=None, probs=None, mean=True, epsilon=1e-9):
    _validate_categorical_params(logits, probs)

    if probs is not None:
        entropy = -np.sum(np.array(probs) * np.log(probs + epsilon), axis=-1)

    if logits is not None:
        logits = log_softmax(logits)
        entropy = -np.sum(np.exp(logits) * logits, axis=-1)

    if mean:
        entropy = np.mean(entropy)
    return entropy


def categorical_sample(logits=None, probs=None, epsilon=1e-9):
    _validate_categorical_params(logits, probs)

    if probs is not None:
        logits = np.log(np.array(probs) + epsilon)

    def gumbel_noise(shape):
        u = np.random.uniform(low=epsilon, high=(1.0 - epsilon), size=shape)
        return -np.log(-np.log(u))

    logits = np.array(logits)
    return np.argmax(logits + gumbel_noise(logits.shape), axis=-1)


class CategoricalEntropy:
    max_abs_value = 1e10

    def pseudorewards(self, policy):
        raise NotImplementedError

    def regularizer(self, policy):
        policy = np.array(policy)
        return np.sum(policy * self.pseudorewards(policy))

    def optimal_policy(self, qualities):
        raise NotImplementedError

    def optimal_value(self, qualities):
        raise NotImplementedError

    def max_regularizer(self, n_actions):
        uniform_policy = np.ones(n_actions) / n_actions
        return self.regularizer(uniform_policy)


@gin.configurable
class ShannonCategoricalEntropy(CategoricalEntropy):

    def pseudorewards(self, policy):
        return -np.log(policy)

    def optimal_policy(self, qualities):
        qualities = np.clip(qualities, -self.max_abs_value, self.max_abs_value)
        return softmax(qualities)

    def optimal_value(self, qualities):
        qualities = np.clip(qualities, -self.max_abs_value, self.max_abs_value)
        return float(log_sum_exp(qualities))


@gin.configurable
class TsallisCategoricalEntropy(CategoricalEntropy):

    @staticmethod
    def k_set(q_values_in_given_state):
        q_values_in_given_state = sorted(q_values_in_given_state, reverse=True)
        return [q for i, q in enumerate(q_values_in_given_state)
                if 1 + (i + 1) * q > sum(q_values_in_given_state[:(i + 1)])]

    def pseudorewards(self, policy):
        return (1 - np.array(policy)) / 2

    def optimal_policy(self, qualities):
        qualities = np.clip(qualities, -self.max_abs_value, self.max_abs_value)

        k = self.k_set(qualities)
        k_set_sum = (sum(k) - 1) / len(k)
        qualities -= k_set_sum

        return np.where(qualities < 0,
                        np.zeros(qualities.shape),
                        qualities)

    def optimal_value(self, qualities):
        qualities = np.clip(qualities, -self.max_abs_value, self.max_abs_value)

        k = self.k_set(qualities)
        k_set_sum = (sum(k) - 1) / len(k)

        return sum([0.5 * (q ** 2) - 0.5 * (k_set_sum ** 2) for q in k]) + 0.5
