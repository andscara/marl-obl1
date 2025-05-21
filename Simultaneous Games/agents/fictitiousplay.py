from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        
        self.count: dict[AgentID, ndarray] = {}

        #
        # TODO: inicializar count con initial si no es None o, caso contrario, con valores random
        #
        if initial is not None:
            self.count = initial
        else:
            for ag in self.game.agents:
                self.count[ag] = np.zeros(shape=self.game.num_actions(ag))

        #
        # TODO: inicializar learned_policy usando de count
        #
        self.learned_policy: dict[AgentID, ndarray] = {}
        for ag in self.game.agents:
            all_count = np.sum(self.count[ag])
            n_actions = self.game.num_actions(ag)
            if all_count == 0:
                self.learned_policy[ag] = np.full(shape=(n_actions), fill_value=1/n_actions)
            else:
                self.learned_policy[ag] = self.count[ag] / all_count

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        #
        # TODO: calcular los rewards de agente para cada acción conjunta 
        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions
        #s
        for joint_action in product(*agents_actions):
            _, joint_action_rewards, _, _, _ = g.step({ g.agents[i]:joint_action[i] for i in range(len(joint_action))})
            rewards[joint_action] = joint_action_rewards[self.agent]

        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        #
        # TODO: calcular la utilidad (valor) de cada acción de agente.
        # Ayuda: iterar sobre rewards para cada acción de agente
        #
        my_agent_index = self.game.agent_name_mapping[self.agent]
        for joint_action in rewards.keys():
            action = joint_action[my_agent_index]

            p = 1.0
            for agent in self.game.agents:
                if agent == self.agent:
                    continue

                agent_index = self.game.agent_name_mapping[agent]
                agent_action = joint_action[agent_index]
                p = p * self.learned_policy[agent][agent_action]

            utility[action] += p * rewards[joint_action]

        return utility
    
    def bestresponse(self):
        utility = self.get_utility()
        #
        # TODO: retornar la acción de mayor utilidad
        #

        # si hay multiples máximos muestreamos
        return self._sampled_argmax(utility)
     
    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()
    
    def policy(self):
       return self.learned_policy[self.agent]
    