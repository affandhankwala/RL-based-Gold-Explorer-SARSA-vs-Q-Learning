## Introduction
This project is part of the Johns Hopkins Reinforcement Learning class number 705.741 and is assigned as the third project. 

## Problem Statement
An agent is given the task of exploring an 8 x 8 matrix by horizontal and vertical actions. This grid is scattered with fixed entities that fill up the entirety of their respective cells within the matrix. These entities include: 
- Mountains: These cells cannot be entered from any direction.
- Mine field: These cells are terminal states that also penalize the agent by the 'LOSE' penalty. 
- Gold coins: These cells reward the agent once entered. They can only be collected once. 
- Motherlode: The winning termial state that rewards the agent the 'WIN' reward. 

As per reinforcement learning principles, each action applies a small penalty upon the agent to accelerate its learning. 

## Methodologies
The two RL based methods that control the agent shall be an off policy Q learning agent and an on policy SARSA agent. These agents will also be tested with various starting positions--a concept known as 'exploring starts'.
To determine which policy performed better, we create 50 agents for each learning agent type and train them a tuned number of episodes. We then perform statistic analysis on the final 50 episodes of each of the 50 agents to draw conclusions on performance and convergence rates. 

See the report for more details. 