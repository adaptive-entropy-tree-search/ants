"""Maximum Entropy Tree Search."""

import enum
import functools

import gin
import numpy as np
from scipy import optimize
from scipy import stats

from alpacka import data
from alpacka import math
from alpacka.agents import stochastic_mcts
from alpacka.agents import tree_search
from alpacka.utils import space as space_utils


@gin.configurable
class SoftQualityNetworkNewLeafRater(stochastic_mcts.NewLeafRater):

    def __init__(
            self, agent, boltzmann_temperature=None, inject_log_temperature=False
    ):
        super().__init__(
            agent,
            use_policy=False,
            boltzmann_temperature=boltzmann_temperature,
        )
        self._inject_log_temperature = inject_log_temperature

    def qualities(self, observation, model):
        del model

        observations = np.expand_dims(observation, axis=0)
        if self._inject_log_temperature:
            log_temperatures = np.array([[np.log(self._agent.temperature)]])
            inp = (observations, log_temperatures)
        else:
            inp = observations

        result = yield inp

        qualities = result[0]
        return qualities

    def network_signature(self, observation_space, action_space):
        obs_sig = space_utils.signature(observation_space)
        if self._inject_log_temperature:
            input_sig = (obs_sig, data.TensorSignature(shape=(1,)))
        else:
            input_sig = obs_sig

        n_actions = space_utils.max_size(action_space)
        action_vector_sig = data.TensorSignature(shape=(n_actions,))
        output_sig = action_vector_sig

        return data.NetworkSignature(input=input_sig, output=output_sig)


class TemperatureTuner:

    def __init__(self, agent):
        self._reference_temperature = agent.reference_temperature
        self._discount = agent.discount

    def __call__(self, root):
        raise NotImplementedError


@gin.configurable
class ConstantTuner(TemperatureTuner):

    def __call__(self, root):
        del root
        return self._reference_temperature


@gin.configurable
class MeanEntropyTuner(TemperatureTuner):

    def __init__(
            self,
            agent,
            target_entropy=1.0,
            min_temperature=0.01,
            max_temperature=10000.0,
    ):
        super().__init__(agent)
        self._target_entropy = target_entropy
        self._min_temperature = min_temperature
        self._max_temperature = max_temperature

    def __call__(self, root):
        qualities = accumulate_qualities(root, self._discount)
        if qualities.size == 0:
            return root.temperature

        def entropy_given_temperature(temperature):
            entropies = math.categorical_entropy(
                logits=(qualities / (temperature + 1e-6)), mean=False
            )
            return np.mean(entropies)

        min_entropy = entropy_given_temperature(self._min_temperature)
        max_entropy = entropy_given_temperature(self._max_temperature)

        def less_or_close(a, b):
            return a < b or np.isclose(a, b)

        if less_or_close(self._target_entropy, min_entropy):
            temperature = self._min_temperature
        elif less_or_close(max_entropy, self._target_entropy):
            temperature = None
        else:
            def excess_entropy(log_temperature):
                return entropy_given_temperature(
                    np.exp(log_temperature)
                ) - self._target_entropy

            log_temperature = optimize.brentq(
                excess_entropy,
                a=np.log(self._min_temperature),
                b=np.log(self._max_temperature),
                rtol=0.01,
            )
            temperature = np.exp(log_temperature)

        return temperature


@gin.configurable
class EntropyRangeTuner(TemperatureTuner):

    def __init__(
            self,
            agent,
            min_entropy=0.1,
            max_entropy=1.0,
            temperature_range=1000.0,
            temperature_penalty=0.001,
    ):
        super().__init__(agent)
        self._min_entropy = min_entropy
        self._max_entropy = max_entropy
        self._min_temperature = agent.reference_temperature / temperature_range
        self._max_temperature = agent.reference_temperature * temperature_range
        self._temperature_penalty = temperature_penalty

    def __call__(self, root):
        qualities = accumulate_qualities(root, self._discount)
        if qualities.size == 0:
            return root.temperature

        def discrepancy(log_temperature):
            temperature = np.exp(log_temperature)
            entropies = math.categorical_entropy(
                logits=(qualities / (temperature + 1e-6)), mean=False
            )
            return np.mean(np.maximum(
                np.maximum(
                    self._min_entropy - entropies,
                    entropies - self._max_entropy,
                ),
                0,
            ) + self._temperature_penalty * log_temperature)

        result = optimize.minimize_scalar(
            discrepancy,
            method='bounded',
            bounds=(
                np.log(self._min_temperature),
                np.log(self._max_temperature),
            ),
        )
        return np.exp(result.x)


@gin.configurable
class StandardDeviationTuner(TemperatureTuner):

    def __init__(self, agent, temperature_range=1000.0):
        super().__init__(agent)
        self._min_temperature = agent.reference_temperature / temperature_range
        self._max_temperature = agent.reference_temperature * temperature_range

    def __call__(self, root):
        qualities = accumulate_qualities(root, self._discount)
        if qualities.size == 0:
            return root.temperature

        std = np.mean(np.std(qualities, axis=-1))

        temperature = std * self._reference_temperature

        return np.clip(
            temperature, self._min_temperature, self._max_temperature
        )


@gin.configurable
class ProbablyApproximatelyOptimalTuner(TemperatureTuner):

    def __init__(
            self,
            agent,
            target_success_rate=0.1,
            tolerance=0.1,
            temperature_range=1000.0,
            min_temperature=0.001,
            retroactive=False,
    ):
        super().__init__(agent)
        self._target_success_rate = target_success_rate
        self._tolerance = tolerance
        self._min_temperature = min_temperature

        self._max_temperature = agent.reference_temperature * temperature_range
        self._retroactive = retroactive

    def __call__(self, root):
        if root.quality(self._discount) is None or (
                self._retroactive and not hasattr(root, 'last_quality')
        ):
            self._store_last_qualities(root)
            return root.temperature

        low = self._return(self._max_temperature, root)
        high = self._return(self._min_temperature, root)

        target = self._tolerance * low + (1 - self._tolerance) * high

        success_rate = functools.partial(
            self._success_rate,
            target=target,
            root=root,
        )

        def less_or_close(a, b):
            return a < b or np.isclose(a, b)

        min_success_rate = success_rate(self._max_temperature)

        min_temperature = self._min_temperature
        max_success_rate = success_rate(self._min_temperature)

        cur_min_temperature = min_temperature
        while (
                less_or_close(max_success_rate, self._target_success_rate) and
                cur_min_temperature < self._max_temperature
        ):
            cur_min_temperature *= 10
            cur_max_success_rate = success_rate(cur_min_temperature)
            if cur_max_success_rate > max_success_rate:
                min_temperature = cur_min_temperature
                max_success_rate = cur_max_success_rate

        if (
                less_or_close(self._target_success_rate, min_success_rate) or
                less_or_close(self._max_temperature, min_temperature)
        ):

            temperature = root.temperature
        elif less_or_close(max_success_rate, self._target_success_rate):
            temperature = min_temperature
        else:
            def excess_success_rate(log_temperature):
                return success_rate(
                    np.exp(log_temperature)
                ) - self._target_success_rate

            log_temperature = optimize.brentq(
                excess_success_rate,
                a=np.log(min_temperature),
                b=np.log(self._max_temperature),
                rtol=0.01,
            )
            temperature = np.exp(log_temperature)

        self._store_last_qualities(root)

        return temperature

    def _retro_is_leaf(self, node):
        if self._retroactive:
            return node.last_was_leaf
        else:
            return node.is_leaf

    def _retro_quality(self, node):
        if self._retroactive:
            return node.last_quality
        else:
            return node.quality(self._discount)

    def _success_rate(self, temperature, target, root):
        def policy(node):
            child_qualities = np.array([
                self._retro_quality(child) for child in node.children
            ])
            return math.softmax(logits=(child_qualities / (temperature + 1e-6)))

        def result(return_):
            return float(return_ >= target)

        return self._fold_return(policy, result, root, level=0, return_so_far=0)

    def _return(self, temperature, root):
        def policy(node):
            child_qualities = np.array([
                child.quality(self._discount) for child in node.children
            ])
            return math.softmax(logits=(child_qualities / (temperature + 1e-6)))

        def result(return_):
            return return_

        return self._fold_return(policy, result, root, level=0, return_so_far=0)

    def _fold_return(self, policy_fn, result_fn, node, level, return_so_far):
        total_discount = self._discount ** level
        if self._retro_is_leaf(node) or node.reward is None:
            total_return = return_so_far + total_discount * node.quality(
                self._discount
            )
            return result_fn(total_return)

        return_including_self = return_so_far + total_discount * node.reward
        child_success_rates = np.array([
            self._fold_return(
                policy_fn, result_fn, child, level + 1, return_including_self
            )
            for child in node.children
        ])
        policy = policy_fn(node)
        return np.sum(policy * child_success_rates)

    def _store_last_qualities(self, node):
        node.last_quality = node.quality(self._discount)
        node.last_was_leaf = node.is_leaf

        for child in node.children:
            self._store_last_qualities(child)


def accumulate_qualities(root, discount):
    def accumulate(node, acc):
        qualities = [
            child.quality(discount)
            for child in node.children
        ]
        if qualities:
            acc.append(qualities)
        for child in node.children:
            accumulate(child, acc)

    acc_qualities = []
    accumulate(root, acc_qualities)
    return np.array(acc_qualities)


@gin.configurable
class SoftIteration:

    def __call__(self, node, discount):
        raise NotImplementedError


@gin.configurable
class SoftPolicyIteration:

    def __init__(self, pseudoreward_shaping=1.0):
        self._pseudoreward_shaping = pseudoreward_shaping

    def __call__(self, node, discount):
        count_sum = sum(child.count for child in node.children)
        policy = np.array([child.count / count_sum for child in node.children])
        pseudorewards = node.categorical_entropy.pseudorewards(policy)

        max_reg = node.categorical_entropy.max_regularizer(len(node.children))
        shift = self._pseudoreward_shaping * max_reg
        pseudorewards = node.temperature * (pseudorewards - shift)

        return sum(
            (child.quality(discount) + pseudoreward) * child.count
            for (child, pseudoreward) in zip(node.children, pseudorewards)
        ) / count_sum


@gin.configurable
class SoftQIteration:

    def __init__(self, pseudoreward_shaping=1.0):
        self._pseudoreward_shaping = pseudoreward_shaping

    def __call__(self, node, discount):
        max_reg = node.categorical_entropy.max_regularizer(len(node.children))
        shift = self._pseudoreward_shaping * max_reg

        return node.temperature * node.categorical_entropy.optimal_value([
            child.quality(discount) / node.temperature - shift
            for child in node.children
        ])


class Node(tree_search.Node):

    def __init__(self, init_quality, temperature, soft_iteration,
                 categorical_entropy):
        super().__init__()

        self._init_quality = init_quality
        self._quality = init_quality
        self._reward_sum = 0
        self._reward_count = 0

        self.temperature = temperature
        self._soft_iteration = soft_iteration
        self.categorical_entropy = categorical_entropy

    def visit(self, reward, value, discount):
        if reward is None:
            return

        self._reward_sum += reward
        self._reward_count += 1

        self.update(discount, value)

    def update(self, discount, value=None):
        if not self.is_leaf:

            value = self.value(discount)
        elif value is None:
            return

        quality_sum = self._reward_sum + discount * value * self._reward_count
        quality_count = self._reward_count

        if self._init_quality is not None:
            quality_sum += self._init_quality
            quality_count += 1

        self._quality = quality_sum / quality_count

    def quality(self, discount):
        del discount
        return self._quality

    @property
    def count(self):
        return self._reward_count + int(self._init_quality is not None)

    def value(self, discount):
        return self._soft_iteration(self, discount)

    @property
    def reward(self):
        if self._reward_count:
            return self._reward_sum / self._reward_count
        else:
            return None


@gin.constants_from_enum
class InitQuality(enum.Enum):
    quality = 0

    log_prior = 1


class TargetPolicy:

    def __call__(self, optimal_policy, node):
        raise NotImplementedError


@gin.configurable
class OptimalTargetPolicy(TargetPolicy):

    def __call__(self, optimal_policy, node):
        del node
        return optimal_policy


@gin.configurable
class MentsTargetPolicy(TargetPolicy):

    def __init__(self, epsilon=1.0):
        self._epsilon = epsilon

    def __call__(self, optimal_policy, node):
        n_actions = len(optimal_policy)
        exploration_decay = np.clip(
            self._epsilon * n_actions / np.log(node.count + 2), 0, 1
        )
        return (
                (1 - exploration_decay) * optimal_policy +
                exploration_decay * np.ones(n_actions) / n_actions
        )


class MaxEntTreeSearchAgent(tree_search.TreeSearchAgent):

    def __init__(
            self,
            new_leaf_rater_class=stochastic_mcts.RolloutNewLeafRater,
            temperature_tuner_class=ConstantTuner,
            soft_iteration_class=SoftPolicyIteration,
            reference_temperature=1.0,
            model_selection_temperature=1e-3,
            real_selection_temperature=1e-3,
            model_selection_tolerance=0.0,
            log_temperature_decay=0.9,
            init_quality=InitQuality.quality,
            target_policy_class=OptimalTargetPolicy,
            n_passes_per_tuning=None,
            categorical_entropy_class=math.ShannonCategoricalEntropy,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._soft_iteration = soft_iteration_class()
        self._reference_temperature = reference_temperature
        self._model_selection_temperature = model_selection_temperature
        self._real_selection_temperature = real_selection_temperature
        self._model_selection_tolerance = model_selection_tolerance
        self._log_temperature_decay = log_temperature_decay
        self._init_quality = init_quality
        self._target_policy = target_policy_class()
        self._n_passes_per_tuning = n_passes_per_tuning or self.n_passes
        self._categorical_entropy = categorical_entropy_class()

        self._tuned_log_temperature = np.log(reference_temperature)
        self._initial_root = None

        self._new_leaf_rater = new_leaf_rater_class(self)
        self._temperature_tuner = temperature_tuner_class(self)

    @property
    def reference_temperature(self):
        return self._reference_temperature

    def _choose_action(self, node, actions, exploratory):
        if exploratory:
            selection_tolerance = self._model_selection_tolerance
            selection_temperature = self._model_selection_temperature
        else:

            selection_tolerance = 1.0
            selection_temperature = self._real_selection_temperature

        qualities = np.array([
            node.children[action].quality(self._discount)
            for action in actions
        ])

        node.temperature = self.temperature
        optimal_policy = self._categorical_entropy.optimal_policy(
            qualities / (node.temperature * selection_temperature)
        )
        target_policy = self._target_policy(optimal_policy, node)

        target_policy = np.clip(target_policy, 0.0, 1.0)
        target_policy /= np.sum(target_policy)
        action = math.categorical_sample(probs=target_policy)
        return actions[action]

    def _after_pass(self, pass_index):
        if (pass_index + 1) % self._n_passes_per_tuning == 0:

            temperature = self._temperature_tuner(self._root)
            if temperature is None:
                temperature = self._root.temperature
            self._update_temperature(temperature)
            self._recalculate_qualities(self._root)

    def _update_temperature(self, temperature):
        n_steps_per_tuning = self._n_passes_per_tuning / self.n_passes

        decay = self._log_temperature_decay ** n_steps_per_tuning

        self._tuned_log_temperature = (
                decay * self._tuned_log_temperature +
                (1 - decay) * np.log(temperature)
        )

    @property
    def temperature(self):
        return np.exp(self._tuned_log_temperature)

    def _recalculate_qualities(self, root):
        def update(node):
            for child in node.children:
                update(child)
            node.temperature = np.exp(self._tuned_log_temperature)
            node.update(self._discount)

        update(root)

    def reset(self, env, observation):
        yield from super().reset(env, observation)
        self._initial_root = self._root
        self._tuned_log_temperature = np.log(self._reference_temperature)

    def _init_root_node(self, state):
        return self._init_node(init_quality=None)

    def _init_child_nodes(self, leaf, observation):
        del leaf
        child_qualities_and_probs = yield from self._new_leaf_rater(
            observation, self._model
        )

        (qualities, prior) = zip(*child_qualities_and_probs)
        if self._init_quality is InitQuality.quality:
            pass
        elif self._init_quality is InitQuality.log_prior:

            qualities = np.array(qualities)
            temp = self._new_leaf_rater._boltzmann_temperature
            value = temp * self._categorical_entropy.optimal_value(qualities / temp)
            qualities = (qualities - value) / temp
        else:
            raise TypeError(
                f'Invalid quality initialization: {self._init_quality}.'
            )

        return list(map(self._init_node, qualities))

    def _init_node(self, init_quality):
        return Node(
            init_quality=init_quality,
            temperature=np.exp(self._tuned_log_temperature),
            soft_iteration=self._soft_iteration,
            categorical_entropy=self._categorical_entropy,
        )

    def network_signature(self, observation_space, action_space):
        return self._new_leaf_rater.network_signature(
            observation_space, action_space
        )

    def _compute_node_info(self, node):
        info = super()._compute_node_info(node)
        softmax_policy = math.softmax(info['qualities'] / node.temperature)
        policy_mismatch = softmax_policy - info['action_histogram']
        return {
            'temperature': node.temperature,
            'softmax_policy': softmax_policy,
            'policy_mismatch': policy_mismatch,
            **info
        }

    @classmethod
    def compute_metrics(cls, episodes):
        metrics = super().compute_metrics(episodes)

        temperatures = np.array([
            temp
            for episode in episodes
            for temp in episode.transition_batch.agent_info['temperature']
        ], dtype=np.float)

        return {
            'temperature_gmean': stats.gmean(temperatures),
            'temperature_min': np.min(temperatures),
            'temperature_max': np.max(temperatures),
            **metrics
        }
