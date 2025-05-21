import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict

class Iql(Agent):
    
    def __init__(
        self,
        game: SimultaneousGame,
        agent: AgentID,
        epsilon_min,
        epsilon_steps,
        lr,
        discount,
        initial=None,
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

        if initial is None:
            self.q = {}
        else:
            self.q = initial

    def _obs_to_tuple(self, obs):
        if obs is None or type(obs) == tuple:
            return obs
        
        # foraging
        if type(obs) == np.ndarray:
            return tuple(obs)
        
        # for the other games the observation is a dictionary with all the actions
        tobs = [None] * self.game.num_agents

        for k, v in obs.items():
            tobs[self.game.agent_name_mapping[k]] = v
        
        return tuple(tobs)

    def action(self):
        self.current_obs = self._obs_to_tuple(self.game.observe(self.agent))
        self.current_action = None

        if self.current_obs is None or (self.learn and np.random.random() < self.epsilon) or self.current_obs not in self.q:
            self.current_action = self.game.action_spaces[self.agent].sample()
            return self.current_action

        self.current_action = self._sampled_argmax(self.q[self.current_obs])

        return self.current_action

    def policy(self):
        return self.q

    def step_update(self):
        if not self.learn:
            return

        next_obs = self._obs_to_tuple(self.game.observe(self.agent))
        r = self.game.reward(self.agent)
        self.epsilon = max(self.epsilon - self.epsilon_update, self.epsilon_min)

        if self.current_obs is None:
            self.current_obs = next_obs
            return

        if next_obs not in self.q:
            self.q[next_obs] = [0] * self.game.num_actions(self.agent)

        if self.current_obs not in self.q:
            self.q[self.current_obs] = [0] * self.game.num_actions(self.agent)

        q = self.q[self.current_obs][self.current_action]

        self.q[self.current_obs][self.current_action] = q + self.lr * (r + self.discount * np.max(self.q[next_obs]) - q)

        self.current_obs = next_obs
