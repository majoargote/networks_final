import numpy as np
from snlearn.message import Message
from typing import List

class Agent:
    def __init__(
            self,
            left_bias: float, 
            right_bias: float, 
            ave_reputation: float, 
            variance_reputation: float, 
            bias_strength: float, 
            reputation_reward_strength: float, 
            reputation_penalty_strength: float,
            forwarding_cost: float = 0.1,
            agent_type: str = 'regular',
            type: str = None,  # 'influencer', 'regular', or 'both'
            ):
        
        # Hidden parameters (used only for sampling distributions)
        self._left_bias = left_bias
        self._right_bias = right_bias
        self._ave_reputation = ave_reputation
        self._variance_reputation = variance_reputation
        
        # Agent parameters
        self.bias_strength = bias_strength
        self.reputation_reward_strength = reputation_reward_strength
        self.reputation_penalty_strength = reputation_penalty_strength
        self.forwarding_cost = forwarding_cost
        
        # Type attribute: 'influencer', 'regular', or 'both'
        self.type = type if type is not None else 'regular'
        
        # Keep agent_type for backward compatibility (maps to old high/low reputation types)
        self.agent_type = agent_type

        # Baseline attributes (sampled from distributions)
        self.bias = self._sample_bias()
        self.baseline_reputation = self._sample_reputation()

        #updating attributes
        self.current_utility = None
        self.current_action = None
        self.reputation = self.baseline_reputation

        #history_storage
        self.utility_history = []
        self.action_history = []
        self.reputation_history = []

    def _sample_bias(self):
        sample = np.random.beta(self._left_bias, self._right_bias)
        bias = 2 * sample - 1
        self.bias = bias
        return bias

    def _sample_reputation(self):
        reputation = np.random.normal(
            self._ave_reputation, 
            np.sqrt(self._variance_reputation))
        return reputation

    def assess_reputation(self, sender_reputation):
        transformed = 1 / (1 + np.exp(-sender_reputation))
        return transformed

    def bias_proximity(self, message_bias):
        proximity = 1 - abs(self.bias - message_bias)
        return proximity

    def estimated_truth(self, message_bias, sender_reputation):
        proximity = self.bias_proximity(message_bias)
        reputation = self.assess_reputation(sender_reputation)
        estimate = proximity * reputation
        return 1 if estimate >= 0.5 else 0
    
    def utility(self, message: Message, sender_reputation: float):
        # Form belief according to model: use truth if observable, otherwise estimated truth
        if message.truth_revealed:
            belief = message.reveal_truth()
        else:
            belief = self.estimated_truth(message.bias, sender_reputation)
        
        proximity = self.bias_proximity(message.bias)
        
        # Calculate gamma = gamma_0 (constant, independent of reputation)
        # and delta(R) = delta_0 * sigma(R) * (1 + sigma(R)) where sigma(R) = 1/(1 + exp(-R))
        # This makes the penalty more aggressive for high-reputation agents
        # R is the agent's own reputation (not sender's)
        # High-reputation agents have much more to lose (higher delta), making them more selective
        gamma_R = self.reputation_reward_strength  # Constant: gamma = gamma_0
        sigma_R = self.assess_reputation(self.reputation)  # sigma(R_i)
        # More aggressive penalty: delta(R) = delta_0 * sigma(R) * (1 + sigma(R))
        # This makes high-reputation agents face super-linear penalties (up to 2x at sigma=1)
        delta_R = self.reputation_penalty_strength * sigma_R * (1 + sigma_R)  # delta(R_i) = delta_0 * sigma(R_i) * (1 + sigma(R_i))
        
        # Expected utility: E[u] = omega*phi + gamma*belief - delta(R)*(1-belief) - cost
        util = (self.bias_strength * proximity 
                + gamma_R * belief 
                - delta_R * (1 - belief)
                - self.forwarding_cost)
        return util
    
    def average_utility(self, message: Message, sender_reputation_list: List[float], store: bool = False):
        utilities = []
        for sender_reputation in sender_reputation_list:
            util = self.utility(message, sender_reputation)
            utilities.append(util)
        avg_util = np.mean(utilities)
        self.current_utility = avg_util
        if store:
            self.utility_history.append(avg_util)
        return avg_util
    
    def decide_action(self, store: bool = False):
        action = 1 if self.current_utility >= 0 else 0
        self.current_action = action
        if store:
            self.action_history.append(action)
        return action
    
    def update_reputation(self, message:Message, store: bool = False):
        if message.truth_revealed:
            message_truth = message.reveal_truth()

            # Reputation update according to model:
            # R_i^{t+1} = R_i^t + gamma * 1{action=1 ∩ truth=1} - delta(R_i^t) * 1{action=1 ∩ truth=0}
            # where gamma = gamma_0 (constant) and delta(R_i^t) = delta_0 * sigma(R_i^t) * (1 + sigma(R_i^t))
            # This makes the penalty more aggressive for high-reputation agents
            # High-reputation agents have much more to lose, making them more selective
            
            if self.current_action == 1 and message_truth == 1:
                # Constant reward: gamma = gamma_0
                gamma_R = self.reputation_reward_strength
                self.reputation += gamma_R
            
            if self.current_action == 1 and message_truth == 0:
                # More aggressive reputation-dependent penalty: delta(R_i^t) = delta_0 * sigma(R_i^t) * (1 + sigma(R_i^t))
                # High-reputation agents face super-linear penalties (up to 2x at sigma=1) when forwarding false messages
                sigma_R = self.assess_reputation(self.reputation)  # sigma(R_i^t)
                delta_R = self.reputation_penalty_strength * sigma_R * (1 + sigma_R)  # delta(R_i^t) = delta_0 * sigma(R_i^t) * (1 + sigma(R_i^t))
                self.reputation -= delta_R

        if store:
            self.reputation_history.append(self.reputation)
        return self.reputation


    