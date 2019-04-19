[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

The project is contained in the Report.ipynb notebook. After installing the required packages (pytorch, unity ml-agents, etc.) clone the git repo and run the iPython notebook. All supporting files (e.g., MADDPG_agent.py) must be downloaded and imported in the notebook.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Unzip and place the contents in the same directory as the iPython notebook.

### Conclusions and References

This environment was solved using various versions of DDPG and MADDPG. First, two separate DDPG agents were used as a baseline. We then used MADDPG, a mult-agent variant of DDPG, to try improve the time it took to solve the environment. Unlike the separate DDPG agents, the MADDPG agents have critics that have access to the other agents' actions and observations. This has been shown to improve both competitive and collaborative agents over separate agents that treat other agents as simply part of the environment. 

Lastly, to try to further improve agents, we add a distributional critic to the MADDPG agents, which learns the entire state-action value distribution (instead of the mean state-action value) using quantile regression. These agents solved the environment faster than the first two variants, illustrating the importance of the critic in guiding training.

The agents in for this project were based on the following papers:
1. DDPG: https://arxiv.org/pdf/1509.02971.pdf
2. MADDPG: https://arxiv.org/pdf/1706.02275.pdf
3. D4PG: https://arxiv.org/pdf/1804.08617.pdf
4. QR-DQN: https://arxiv.org/pdf/1710.10044.pdf
