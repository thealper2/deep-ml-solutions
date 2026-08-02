import numpy as np

class PettingZooToGymAdapter:
    """Adapter that wraps a turn-based multi-agent environment
    into a Gym-style interface with macro-step semantics."""
    
    def __init__(self, env):
        """Initialize the adapter with a PettingZoo-style environment."""
        self.env = env
        self.agents = []
        self.agent_rewards = {}
        self.agent_dones = {}
        self._initialized = False
    
    def reset(self):
        """Reset the environment and return observations for all agents."""
        self.env.reset()
        self.agents = list(self.env.agents)
        self.agent_rewards = {agent: 0.0 for agent in self.agents}
        self.agent_dones = {agent: False for agent in self.agents}
        self._initialized = True
        
        observations = {agent: self.env.observe(agent) for agent in self.agents}
        return observations
    
    def step(self, actions: dict):
        """Execute one macro-step (all agents act once in order).
        
        Args:
            actions: dict mapping agent names to their actions
            
        Returns:
            Tuple of (observations, total_reward, all_done, info)
        """
        if not self._initialized:
            raise RuntimeError("Environment must be reset before stepping")
        
        self.agent_rewards = {agent: 0.0 for agent in self.agents}
        
        for agent in self.agents:
            if self.agent_dones.get(agent, False):
                continue
            
            self.env.step(actions[agent])
            
            if agent in self.env.rewards:
                self.agent_rewards[agent] = self.env.rewards[agent]
            
        for agent in self.agents:
            done = False
            if agent in self.env.terminations:
                done = done or self.env.terminations[agent]
            
            if agent in self.env.truncations:
                done = done or self.env.truncations[agent]
            
            self.agent_dones[agent] = done
        
        observations = {agent: self.env.observe(agent) for agent in self.agents}
        
        total_reward = round(sum(self.agent_rewards.values()), 4)
        
        all_done = all(self.agent_dones.get(agent, True) for agent in self.agents)
        
        info = {
            'agent_rewards': self.agent_rewards.copy(),
            'agent_dones': self.agent_dones.copy()
        }
        
        return observations, total_reward, all_done, info
    
    def get_agents(self):
        """Return the list of agent names."""
        return self.agents.copy()
