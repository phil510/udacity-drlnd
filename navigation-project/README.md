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

### Getting Started

The project is contained in the Report.ipynb notebook. After installing the required packages (pytorch, unity ml-agents, etc.) clone the git repo and run the iPython notebook. All supporting files (dqn.py, replay_buffers.py, etc.) must be downloaded and imported in the notebook.

Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
Unzip and place the environment in the same directory as the iPython notebook.

### Conclusions and References

Multiple agent designs were used to solve the envirnment, including a vanilla DQN, a double DQN agent, a dueling DQN agent, a prioritized DQN agent, and a rainbow DQN (name borrowed from the larger rainbow agent described in the paper below). The fastest learner was able to solve the environment in less than 400 episodes. While all agents were able to solve the environment efficiently, certain agents learned faster and the double and prioritized flavors seemed to provide the largest improvement over the vanilla DQN.

The agents in for this project were based on the following papers:
1. DQN: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
2. Double DQN: https://arxiv.org/abs/1509.06461
3. Dueling DQN: https://arxiv.org/abs/1511.06581
4. DQN with Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
5. Rainbow DQN: https://arxiv.org/pdf/1710.02298

The sum-tree implementation of prioritized experience replay uses the OpenAI baselines sum tree code, which can be found here: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py and is included segment_tree.py in the git repo.
