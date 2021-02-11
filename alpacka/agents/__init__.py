import gin

from alpacka.agents import core
from alpacka.agents import dummy
from alpacka.agents import mets
from alpacka.agents import models
from alpacka.agents import stochastic_mcts
from alpacka.agents.base import *


def configure_agent(agent_class):
    return gin.external_configurable(
        agent_class, module='alpacka.agents'
    )


ActorCriticAgent = configure_agent(core.ActorCriticAgent)
PolicyNetworkAgent = configure_agent(core.PolicyNetworkAgent)
RandomAgent = configure_agent(core.RandomAgent)

MaxEntTreeSearchAgent = configure_agent(mets.MaxEntTreeSearchAgent)
StochasticMCTSAgent = configure_agent(stochastic_mcts.StochasticMCTSAgent)

PerfectModel = configure_agent(models.PerfectModel)


class _DistributionAgent:

    def __init__(self, distribution, with_critic, parameter_schedules):
        super().__init__()

        if with_critic:
            self._agent = ActorCriticAgent(
                distribution, parameter_schedules=parameter_schedules)
        else:
            self._agent = PolicyNetworkAgent(
                distribution, parameter_schedules=parameter_schedules)

    def __getattr__(self, attr_name):
        return getattr(self._agent, attr_name)
