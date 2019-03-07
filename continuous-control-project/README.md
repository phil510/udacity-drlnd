[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, we worker with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To speed up training, we used a distributed environment that contains 20 identical agents, each with its own copy of the environment.   

To solve the environment, the agent must get an average score of +30 over 100 consecutive episodes and over all agents.  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).
The environment is considered solved, when the average over 100 episodes of those average scores is at least +30. 

### Getting Started

The project is contained in the Report.ipynb notebook. After installing the required packages (pytorch, unity ml-agents, etc.) clone the git repo and run the iPython notebook. All supporting files (e.g., PPO_agent.py) must be downloaded and imported in the notebook.

Next, download the environment from one of the links below.  You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Unzip and place the environment in the same directory as the iPython notebook.

### Conclusions and References

This environment was solved with a PPO agent. Like many environments, a significant amount of hyperparameter tuning was required to efficiently solve the environment within a reasonable time frame. Small, separate deep neural networks were used for value estimation and policy estimation - no shared layers were utilized. Consistent with the original PPO paper (reference below), a truncated version of generalized advantage estimation (GAE) was used. However, the advantage function used by A3C, which is equivalent to GAE with lambda equal to 1.0, was also tried during hyperparameter tuning with some success. 

The number of steps between updates, however, was probably the most sensitive parameter. Using a number of steps above 500, which corresponds to about 2 updates per episode, proved to be much more robust than smaller step counts; agents with step counts of 256 and 128 could not consistently solve the environment within even 500 episodes.

The agent in for this project was based on the following papers:
1. PPO: https://arxiv.org/pdf/1707.06347.pdf
2. Generalized Advantage Estimation: https://arxiv.org/pdf/1506.02438.pdf
3. A3C: https://arxiv.org/pdf/1602.01783.pdf
