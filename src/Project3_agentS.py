"""
This class serves as the model of an agent capable of navigating an environment via SARSA
"""
from Project3_env import GoldExplorer
import random 

class SARSA_0:
    def __init__(self, gold_explorer: GoldExplorer, alpha: float, discount_rate: float):
        # Assign environment
        self.environment = gold_explorer
        # Assign learning rate
        self.learning_rate = alpha
        # Assign discount factor
        self.discount_rate = discount_rate
        # Initialize q table
        self.q_table = {}
        
    

    """
    This function determines the best action and it's associated q value for any particular state action pair
    """
    def best_q_and_a (self, state: int) -> list: 
        # Check if state exists in q table
        if state in self.q_table: 
            # Initialize stored best values
            best_q, best_a = -9999, None
            # Determine which action has the highest q value
            for action, q_val in self.q_table[state].items():
                if q_val > best_q: 
                    best_q = q_val
                    best_a = action
            # Return best values
            return best_q, best_a
        # Return random action
        return 0, random.choice([0, 1, 2, 3])

    """
    train() function trains the q table by traversing the agent through the environment over a set number of episodes.
    Parameters:
    episodes: Number of training episodes (each episode is from agent initialization till finish)
    exp_starts: Boolean indicating whether to explore starts.
    """
    def train(self, episodes: int, exp_starts: bool=False) -> list:
        # Initialize expontential decay e values
        e_max, e_min = 1, -0.02
        # Initialize Episode rewards list
        episode_rewards = []
        for ep in range(episodes):
            # Update E
            r = max((episodes - ep) / episodes, 0)
            e = max((e_max - e_min) * r + e_min, 0.01)
            # Reset the environment
            self.environment.reset(exp_starts)
            state = self.environment.get_state()
            # Initialize a state, actions, and rewards list
            states, actions, rewards = [state], [], []
            # Initialize episode reward
            # Initialize boolean indicating whether game is finished
            game_end = False

            # Iterate until game terminates
            while not game_end:
                # Determine if agent explores or exploits
                decision = random.choices(['exploit', 'explore'], weights=[1 - e, e])
                # Select an action
                if decision[0] == 'exploit': 
                    # Get the best action
                    best_q_and_a = self.best_q_and_a(state)
                    action = best_q_and_a[1]
                else: 
                    action = random.choice([0, 1, 2, 3])
                # Act on the action
                state, reward, game_end = self.environment.execute_action(action)
                # Store state, actions, and rewards
                states.append(state)
                actions.append(action)
                rewards.append(reward)
            # Once agent, reaches goal, update q table
            for i in range(len(states) - 1):
                # Pull current state, next state, action, next action, and rewards
                s = states[i]
                next_s = states[i + 1]
                a = actions[i]
                r = rewards[i]

                # Retrieve old Q value
                q = 0
                # Determine if current state has been explored and has old state value
                if s in self.q_table: 
                    # Determine if current action has been explored
                    if a in self.q_table[s]:
                        # Pull q value from this state, action pair
                        q = self.q_table[s][a]
                # Determine next state q value
                next_q = 0
                # Determine if next_state is in q table
                if next_s in self.q_table:
                    # Retrieve next action if next state is not terminal state
                    if i < len(states) - 2:
                        next_a = actions[i + 1]
                        # Pull q from next_state, next_action pair
                        if next_a in self.q_table[next_s]:
                            next_q = self.q_table[next_s][next_a]
                # Calculate updated q value
                q = q + self.learning_rate * (r + self.discount_rate * next_q - q)

                # Update q table 
                # Determine if state exists in q_table
                if s in self.q_table:
                    self.q_table[s][a] = q
                else:
                    # Create a state dictionary
                    self.q_table[s] = {}
                    # Assign q value to state action pair
                    self.q_table[s][a] = q
            # Sum all episode rewards 
            episode_rewards.append(sum(rewards))
        # Return episode_rewards
        return episode_rewards


