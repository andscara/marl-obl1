import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict
from numpy import ndarray
from itertools import product

class Jalam(Agent):
    
    def __init__(
        self,
        game: SimultaneousGame,
        agent: AgentID,
        epsilon_min,
        epsilon_steps,
        lr,
        discount,
        seed=None
    ) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        
        self.learn = True
        epsilon_max = 1
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_update = (epsilon_max - epsilon_min) / epsilon_steps
        self.lr = lr
        self.discount = discount

        self._products = []

        self.q = {}

        self.learned_policy: dict[AgentID, dict[tuple, ndarray]] = {}
        self.count: dict[AgentID, dict[tuple, ndarray]] = {} 
        for ag in self.game.agents:
            if ag != self.agent:
                self.count[ag] = {}
                self.learned_policy[ag] = {}

    def update_models(self) -> None:
        if callable(self.game.observe_action):
            # foraging
            joint = self.game.observe_action(self.agent)
            actions = {k: joint[self.game.agent_name_mapping[k]] for k in self.game.agents}
        else:
            # other games
            actions = self.game.observe(self.agent)

        if actions is None or self.current_obs is None:
            return

        for agent in self.game.agents:
            if agent != self.agent:
                if self.current_obs not in self.count[agent]:
                    self.count[agent][self.current_obs] = np.zeros(self.game.num_actions(agent))
                
                self.count[agent][self.current_obs][actions[agent]] += 1
                
                #if self.current_obs not in self.learned_policy[agent]:
                #    self.learned_policy[agent][self.current_obs] = np.zeros(self.game.num_actions(agent))
                
                self.learned_policy[agent][self.current_obs] = self.count[agent][self.current_obs] / np.sum(self.count[agent][self.current_obs])

    def _obs_to_tuple(self, obs):
        if obs is None or type(obs) == tuple:
            return obs

        # foraging
        if type(obs) == np.ndarray:
            return tuple(obs)
        
        tobs = [None] * self.game.num_agents

        for k, v in obs.items():
            tobs[self.game.agent_name_mapping[k]] = v
        
        return tuple(tobs)

    def get_action_values(self, obs):
        if len(self._products) == 0:
            agents_actions = list(map(lambda agent: list(self.game.action_iter(agent)), self.game.agents))
            self._products = list(product(*agents_actions))

        av = np.zeros(self.game.num_actions(self.agent))

        if obs not in self.q:
            return av

        my_index = self.game.agent_name_mapping[self.agent]
        for joint_action in self._products:
            action = joint_action[my_index]

            if joint_action not in self.q[obs]:
                continue

            p = 1.0
            for agent in self.game.agents:
                if agent == self.agent:
                    continue

                agent_index = self.game.agent_name_mapping[agent]
                agent_action = joint_action[agent_index]

                if obs not in self.learned_policy[agent]:
                    p = p * (1 / self.game.num_actions(self.agent))
                else:
                    p = p * self.learned_policy[agent][obs][agent_action]
            
            av[action] += p * self.q[obs][joint_action]
        
        return av

    def action(self):
        self.current_obs = self._obs_to_tuple(self.game.observe(self.agent))
        self.current_action = None

        if (self.current_obs is None or
            (self.learn and np.random.random() < self.epsilon) or
            self.current_obs not in self.q):
            self.current_action = self.game.action_spaces[self.agent].sample()
            return self.current_action

        av = self.get_action_values(self.current_obs)

        self.current_action = self._sampled_argmax(av)

        return self.current_action

    def policy(self):
        return self.q

    def get_joint_action(self, obs):
        if callable(self.game.observe_action):
            return self.game.observe_action(self.agent)
        return obs

    def step_update(self):
        if not self.learn:
            return

        next_obs = self._obs_to_tuple(self.game.observe(self.agent))
        r = self.game.reward(self.agent)
        self.epsilon = max(self.epsilon - self.epsilon_update, self.epsilon_min)

        if self.current_obs is None:
            self.current_obs = next_obs
            return

        self.update_models()

        if next_obs not in self.q:
            self.q[next_obs] = {}

        if self.current_obs not in self.q:
            self.q[self.current_obs] = {}
        
        joint_action = self.get_joint_action(next_obs)

        if joint_action not in self.q[self.current_obs]:
            self.q[self.current_obs][joint_action] = 0

        q = self.q[self.current_obs][joint_action]

        av = self.get_action_values(next_obs)
        self.q[self.current_obs][joint_action] = q + self.lr * (r + self.discount * np.max(av) - q)

        self.current_obs = next_obs
