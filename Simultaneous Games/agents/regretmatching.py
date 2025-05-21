import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict

class RegretMatching(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        if (initial is None):
          self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        else:
          self.curr_policy = initial.copy()
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
        np.random.seed(seed=seed)

    def regrets(self, played_actions: ActionDict) -> np.ndarray:
        actions = played_actions.copy()
        u = np.zeros(self.game.num_actions(self.agent), dtype=float)
        reward = self.game.rewards[self.agent]
        #
        # TODO: calcular regrets
        #
        for alt_action in range(len(u)):
            g = self.game.clone()
            actions[self.agent] = alt_action
            g.step(actions)
            alt_reward = g.rewards[self.agent]
            u[alt_action] = alt_reward - reward
        
        return u
    
    def regret_matching(self):
        #
        # TODO: calcular curr_policy y actualizar sum_policy
        #
        adj_regrets = np.array(list(map(lambda r: max(r,0), self.cum_regrets)))
        s = np.sum(adj_regrets)
        if s == 0:
            n_actions = self.game.num_actions(self.agent)
            self.curr_policy = np.full(shape=n_actions, fill_value=1/n_actions)
        else:
            self.curr_policy = adj_regrets / np.sum(adj_regrets)

        self.sum_policy += self.curr_policy

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
           return
        regrets = self.regrets(actions)
        self.cum_regrets += regrets
        self.regret_matching()
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1))
    
    def policy(self):
        return self.learned_policy
