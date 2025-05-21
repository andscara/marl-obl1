from base.game import SimultaneousGame, AgentID, ActionDict
import gymnasium as gym
from lbforaging.foraging.environment import ForagingEnv, Player, Action

class Foraging(SimultaneousGame):
    def __init__(self, config: str | None = None, seed: int | None = None):
    
        self.config = config
        # environment
        if self.config is None:
            self.config = "Foraging-8x8-2p-1f-v3"
        self.env = gym.make(self.config)

        # action set
        self.action_set = [a.name for a in list(Action)]

        # seed
        self.seed = seed

        # agents
        self.agents = ["agent_" + str(r) for r in range(self.env.unwrapped.n_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents()))))

        self.observations = None
        self.rewards = None
        self.terminations = None
        self.truncations = None
        self.infos = None

        # actions
        self.action_spaces = {
            agent: self.env.action_space[i] for i, agent in enumerate(self.agents)
        }

    # num_agents
    def num_agents(self):
        return len(self.agents)
    
    # step
    def step(self, actions: ActionDict) -> tuple[dict, dict, dict, dict, dict]:
        # actions
        action = []
        for agent in self.agents:
            action.append(actions[agent])
        action = tuple(action)

        # step
        obs, rewards, done, truncated, info = self.env.step(action=action)
        self.current_step = self.env.unwrapped.current_step

        # update observations, rewards, terminations, truncations, infos
        for i, agent in enumerate(self.agents):
            self.rewards[agent] = rewards[i]
            self.observations[agent] = {
                'observation': obs[i].copy(),   # observation of the agent
                'action': action                # joint action of all agents
            }
            self.terminations[agent] = done
            self.truncations[agent] = truncated
            self.infos[agent] = info
        
        self._done = done
        self._truncated = truncated
        
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def render_start_array(self):
        env = gym.make(self.config)
        env.reset(seed=self.seed)
        image = env.unwrapped.render(mode="rgb_array")
        env.unwrapped.close()
        
        return image

    # reset
    def reset(self):
        # Recreate so that it reset starts in the same position
        self.env = gym.make(self.config)
        obs, _ = self.env.reset(seed=self.seed)

        self.observations = dict(map(lambda agent: (agent, {'observation': obs[self.agent_name_mapping[agent]], 'action': None}), self.agents))
        self.rewards = dict(map(lambda agent: (agent, 0), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))
        self._done = False
        self._truncated = False
    
    # get observation
    def observe(self, agent: AgentID):
        # check if agent is valid
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not valid. Valid agents are: {self.agents}")
        # get observation
        observation = self.observations[agent]['observation']
        return observation
    
    # get actions
    def observe_action(self, agent: AgentID):
        # check if agent is valid
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not valid. Valid agents are: {self.agents}")
        # get action
        action = self.observations[agent]['action']
        return action
    
    # render
    def render(self):
        self.env.render()

    # close
    def close(self):
        self.env.close()

    # done
    def done(self):
        return self._done
