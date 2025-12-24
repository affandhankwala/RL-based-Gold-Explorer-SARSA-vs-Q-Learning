from Project3_agentQ import QAgent
from Project3_agentS import SARSA_0
from Project3_env import GoldExplorer
import matplotlib.pyplot as plt
import numpy as np


"""
This method trains n agents of the same type and gathers the return per episode. We also calculate the mean return for each episode and 
return the mean of these means of the last 50 episodes.
"""
def train_n_agents(agent_count: int, agent_type: str, exp_starts: bool = False) -> list:
    # Instantiate parameters
    environment = GoldExplorer()
    alpha = 0.1
    discount = 0.95
    episode_count = 2000
    total_g_n = []
    # Train n agents
    for i in range(agent_count):
        # Determine which agent to train
        if agent_type == "QAgent":
            agent = QAgent(environment, alpha, discount)
        else:
            agent = SARSA_0(environment, alpha, discount)
        # Train the agent
        g_n = agent.train(episode_count, exp_starts)
        # Store g_n
        total_g_n.append(g_n)
    # Calculate the mean and standard error among all agents for each episode, g_bar_n
    g_bar_n = []
    g_se_n = []
    for episode in range(len(total_g_n[0])):
        episode_g_n = []
        for agent in range(len(total_g_n)):
            episode_g_n.append(total_g_n[agent][episode])
        # Append mean 
        g_bar_n.append(np.mean(episode_g_n))
        # Append standard error as standard dev / sqrt(n)
        g_se_n.append(np.std(episode_g_n, ddof=1) / np.sqrt(len(episode_g_n)))
    # Calculate the mean of trained agents as mean of last 50 episodes
    g_bar_list = []
    for i in range(50):
        g_bar_list.append(g_bar_n[len(g_bar_n) - 1 - i])
    g_bar = np.mean(g_bar_list)
    g_med = np.median(g_bar_list)
    # Return average per episode, se per episode, and mean of last 50 episodes
    return g_bar_n, g_se_n, g_bar, g_med
    
"""
This method is responsible for plotting the mean and se data as subplots
"""
def plot_graph(title: str, mean_data: list, se_data):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    # Mean scatterplot
    mean_x = list(range(0, len(mean_data)))
    axes[0].scatter(mean_x, mean_data, color="b")
    axes[0].set_title(f"{title}: Means")
    axes[0].legend()
    # SE scatterplot
    se_x = list(range(0, len(se_data)))
    axes[1].scatter(se_x, se_data, color="r")
    axes[1].set_title(f"{title}: Standard Error")
    axes[1].legend()

    plt.tight_layout()
    # Save the plot
    plt.savefig(title, dpi=300, bbox_inches='tight')

def main():
    # Train 50 q Learning agents
    q_g_bar_n, q_g_se_n, q_g_bar, q_g_med = train_n_agents(50, "QAgent")
    print("50 QAgents trained")
    # Train 50 SARSA Learning agents
    s_g_bar_n, s_g_se_n, s_g_bar, s_g_med = train_n_agents(50, "SARSA")
    print("50 SARSA agents trained")
    # Train 50 q Learning agents with exploring starts
    q_g_bar_n_exp, q_g_se_n_exp, q_g_bar_exp, q_g_med_exp = train_n_agents(50, "QAgent", True)
    print("50 QAgents trained while exploring starts")
    # Train 50 SARSA Learning agents with exploring starts
    s_g_bar_n_exp, s_g_se_n_exp, s_g_bar_exp, s_g_med_exp = train_n_agents(50, "SARSA", True)
    print("50 SARSA agents trained with exploring starts")
    
    # Plot metrics
    plot_graph("50 Q Agents without Exploring Starts", q_g_bar_n, q_g_se_n)
    plot_graph("Last 50 ep of 50 Q Agents without Exploring Starts", q_g_bar_n[-50:], q_g_se_n[-50:])
    print("50 Q Agents w/o Exp Starts G_Bar: ", q_g_bar)
    print("50 Q Agents w/o Exp Starts G_Med: ", q_g_med)
    plot_graph("50 SARSA Agents without Exploring Starts", s_g_bar_n, s_g_se_n)
    plot_graph("Last 50 ep of 50 SARSA Agents without Exploring Starts", s_g_bar_n[-50:], s_g_se_n[-50:])
    print("50 SARSA Agents w/o Exp Starts G_Bar: ", s_g_bar)
    print("50 SARSA Agents w/o Exp Starts G_Med: ", s_g_med)
    plot_graph("50 Q Agents with Exploring Starts", q_g_bar_n_exp, q_g_se_n_exp)
    plot_graph("Last 50 ep of 50 Q Agents with Exploring Starts", q_g_bar_n_exp[-50:], q_g_se_n_exp[-50:])
    print("50 Q Agents w/ Exp Starts G_Bar: ", q_g_bar_exp)
    print("50 Q Agents w/ Exp Starts G_Med: ", q_g_med_exp)
    plot_graph("50 SARSA Agents with Exploring Starts", s_g_bar_n_exp, s_g_se_n_exp)
    plot_graph("Last 50 ep of 50 SARSA Agents with Exploring Starts", s_g_bar_n_exp[-50:], s_g_se_n_exp[-50:])
    print("50 SARSA Agents w/ Exp Starts G_Bar: ", s_g_bar_exp)
    print("50 SARSA AGents w/ Exp Starts G_Med: ", s_g_med_exp)
    

main()