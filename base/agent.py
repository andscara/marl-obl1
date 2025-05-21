from base.game import SimultaneousGame, AgentID
import numpy as np

class Agent():

    def __init__(self, game:SimultaneousGame, agent: AgentID) -> None:
        self.game = game
        self.agent = agent
    
    # Replace numpy argmax
    def _sampled_argmax(self, x):
        v = np.max(x)
        filter = (x == v).nonzero()[0]

        return np.random.choice(filter)

    def action(self):
        pass

    def policy(self):
        pass

    def step_update(self):
        pass
    
    def reset(self):
        pass