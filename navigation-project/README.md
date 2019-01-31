[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, an agent was trained to navigate and collect bananas in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and, in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

Multiple agent designs were used to solve the envirnment, including a vanilla DQN, a double DQN agent, a dueling DQN agent, a prioritized DQN agent, and a rainbow DQN (name borrowed from the larger rainbow agent described in the paper below). While all agents were able to solve the environment efficiently, certain agents learned faster and the double and prioritized flavors seemed to provide the largest improvement over the vanilla DQN.

The agents in for this project were based on the following papers:
1. DQN: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
2. Double DQN: https://arxiv.org/abs/1509.06461
3. Dueling DQN: https://arxiv.org/abs/1511.06581
4. DQN with Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
5. Rainbow DQN: https://arxiv.org/pdf/1710.02298
