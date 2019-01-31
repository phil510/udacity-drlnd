import numpy as np
import gym
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_buffers import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent():
    def __init__(self, model, 
                 model_params, 
                 state_processor, 
                 n_actions,
                 gamma = 0.99,
                 epsilon = 1.0,
                 min_epsilon = 1e-2,
                 epsilon_decay = .999,
                 loss_function = F.smooth_l1_loss, 
                 optimizer = optim.Adam,
                 learning_rate = 1e-3,
                 l2_regularization = 0.0,
                 batch_size = 32,
                 replay_memory = 1000000,
                 replay_start = 50000,
                 target_update_freq = 1000,
                 action_repeat = 4,
                 update_freq = 4, 
                 random_seed = None):
        '''
        DQN Agent from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
        
        model: pytorch model class
            callable model class for the online and target networks
            the DQN agent instantiates each model
            
        model_params: dict
            dictionary of parameters used to define the model class (e.g., feature 
            space size, action space size, etc.)
            this should be the only input to instantiate the model class
            
        state_processor: function
            callable function that takes state as the input and outputs the processed state
            to use as a feature for the model
            the processed states are stored as experiences in the replay buffer
            
        n_actions: int
            the number of actions the agent can perform
            
        gamma: float, [0.0, 1.0]
            discount rate parameter
            
        epsilon: float, [0.0, 1.0]
            epsilon used to compute the epsilon-greedy policy
            
        min_epsilon: float, [0.0, 1.0]
            minimun value for epsilon over all episodes
            
        epsilon_decay: float, (0.0, 1.0]
            rate at which to decay epsilon after each episodes
            1.0 corresponds to no decay
            
        loss_function: pytorch loss (usually the functional form)
            callable loss function that takes inputs, targets as positional arguments
            
        optimizer: pytorch optimizer
            callable optimizer that takes the learning rate as a parameter
            
        learning_rate: float
            learning rate for the optimizer
            
        l2_regularization: float
            hyperparameter for L2 regularization
            
        batch_size: int
            batch size parameter for training the online network
            
        replay_memory: int
            maximum size of the replay memory
            
        replay_start: int
            number of actions to take/experiences to store before beginning to train 
            the online network 
            this should be larger than the batch size to avoid the same experience
            showing up multiple times in the batch
            
        target_update_freq: int
            the frequency at which the target network is updated with the online
            network's weights
            
        action_repeat: int
            the number of times to repeat the same action
            
        update_freq: int
            the number of steps between each SGD (or other optimization) update
            
        seed: None or int
            random seed for the replay buffer
        '''
        self.n_actions = n_actions
        self.actions = np.arange(self.n_actions)
        
        self.state_processor = state_processor
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_start = replay_start
        self.target_update_freq = target_update_freq
        self.action_repeat = action_repeat
        self.update_freq = update_freq
        
        self.reset_current_step()
        
        self.replay_buffer = ReplayBuffer(memory_size = replay_memory, seed = random_seed)
        
        self.online_network = model(model_params)
        self.target_network = model(model_params)
        self.assign_variables()
        
        self.loss_function = loss_function
        self.optimizer = optimizer(self.online_network.parameters(), 
                                   lr = learning_rate, 
                                   weight_decay = l2_regularization)
    
    def assign_variables(self):
        '''
        Assigns the variables (weights and biases) of the online network to the target networl
        '''
        self.target_network.load_state_dict(self.online_network.state_dict())
        
    def reset_current_step(self):
        '''
        Set the current_step attribute to 0
        '''
        self.current_step = 0
    
    def process_state(self, state):
        '''
        Process the state provided by the environment into the feature used by the 
        online and target networks
        
        state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
        '''
        processed_state = self.state_processor(state)
        
        return processed_state
    
    def add_to_replay_memory(self, state, action, reward, next_state, terminal):
        '''
        Add the state, action, reward, next_state, terminal tuple to the replay buffer
        
        state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
            
        action: int, provided by the environment
            index of the action taken by the agent
            
        reward: float, provided by the environment
            reward for the given state, action, next state transition
            
        next_state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
        
        terminal: bool, usually provided by the environment
            whether or not the current episode has ended
        '''
        processed_state = self.process_state(state)
        processed_next_state = self.process_state(next_state)
        
        experience = (processed_state, action, reward, processed_next_state, terminal)
        self.replay_buffer.add(experience)
    
    def action(self, state, mode = 'train'):
        '''
        Selects an action according to the greedy or epsilon-greedy policy
        
        state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
            
        mode: 'train' or 'test'
            selects an action acording to the epsilon-greedy policy when set to 'train'
            selects an action acording to the greedy policy when set to 'test'
        '''
        if (self.current_step % self.action_repeat == 0) or (not hasattr(self, 'previous_action')):
            if mode == 'test':
                state_policy, action = self.greedy_policy(state)
            else:
                state_policy, action = self.epsilon_greedy_policy(state)
        else:
            action = self.previous_action
                
        self.previous_action = action

        return action
    
    def greedy_policy(self, state):
        '''
        Returns the greedy policy as a discrete probability distribution and the 
        greedy action
        All actions except the greedy action have probablity 0
        
        state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
        '''
        Q_s = self.estimate_q(state, process_state = True)
        
        action = np.argmax(Q_s)
        policy = np.zeros(self.n_actions)
        policy[action] = 1.0
        
        return policy, action
    
    def epsilon_greedy_policy(self, state):
        '''
        Returns the epsilon-greedy policy as a discrete probability distribution and
        an action randomly selected according to the probability distribution
        
        state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
        '''
        Q_s = self.estimate_q(state, process_state = True)
        
        policy = np.ones(self.n_actions) * self.epsilon / self.n_actions
        policy[np.argmax(Q_s)] = 1.0 - self.epsilon + self.epsilon / self.n_actions
        
        action = np.random.choice(self.actions, p = policy)
        
        return policy, action
    
    def estimate_q(self, state, process_state = True):
        '''
        Estimates the Q values for a given state and all actions from the online network
        
        state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
            
        process_state: bool
            whether to process the state before estimating Q_s
        '''
        if process_state:
            processed_state = self.process_state(state)
        else: 
            processed_state = state
            
        with torch.no_grad():
            Q_s = self.online_network(processed_state)
        
        return Q_s
        
    def estimate_target_q(self, state, process_state = True):
        '''
        Estimates the Q values for a given state and all actions from the target network
        
        state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
            
        process_state: bool
            whether to process the state before estimating Q_s
        '''
        if process_state:
            processed_state = self.process_state(state)
        else: 
            processed_state = state
            
        with torch.no_grad():
            Q_s = self.target_network(processed_state)
        
        return Q_s
    
    def update_target(self, state, action, reward, next_state, terminal, process_state = True):
        '''
        Calculates the update target for the state, action, reward, next_state, terminal tuple
        
        state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
            
        action: int, provided by the environment
            index of the action taken by the agent
            
        reward: float, provided by the environment
            reward for the given state, action, next state transition
            
        next_state: object, provided by the environment
            state provided by the environment, usually a vector or tensor
        
        terminal: bool, usually provided by the environment
            whether or not the current episode has ended
            
        process_state: bool
            whether to process the state before estimating Q_s_next
        '''
        Q_s_next = self.estimate_target_q(next_state, process_state = process_state)
        terminal_mask = torch.tensor([not t for t in terminal], dtype = torch.float32)
        update_target = reward + self.gamma * torch.max(Q_s_next, dim = 1)[0] * terminal_mask
        
        return update_target
    
    def update(self):
        '''
        Updates the model by taking a step from the optimizer
        The version does not include gradient clipping
        '''
        if self.current_step >= self.replay_start:
            if self.current_step % self.target_update_freq == 0:
                self.assign_variables()
                
            if self.current_step % self.update_freq == 0:
                experiences = self.replay_buffer.sample(self.batch_size)
                state, action, reward, next_state, terminal = zip(*experiences)
                
                state = torch.cat(state)
                action = torch.tensor(action, dtype = torch.int64)
                reward = torch.tensor(reward, dtype = torch.float32)
                next_state = torch.cat(next_state)
                
                update_target = self.update_target(state, action, reward, next_state, terminal, process_state = False)
                Q_sa = self.online_network(state).gather(1, action.unsqueeze(1)).squeeze()
                
                loss = self.loss_function(Q_sa, update_target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
        self.current_step += 1

    def update_epsilon(self):
        '''
        Decays epsilon by the decay rate
        '''
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
class DoubleDQNAgent(DQNAgent):
    def update_target(self, state, action, reward, next_state, terminal, process_state = True):
        Q_s_next = self.estimate_q(next_state, process_state = process_state)
        max_action = torch.max(Q_s_next, dim = 1)[1]
        
        Q_s_next = self.estimate_target_q(next_state, process_state = process_state)
        Q_max = Q_s_next.gather(1, max_action.unsqueeze(1)).squeeze()

        terminal_mask = torch.tensor([not t for t in terminal], dtype = torch.float32)
        update_target = reward + self.gamma * Q_max * terminal_mask
        
        return update_target
        
class PrioritizedDQNAgent(DQNAgent):
    def __init__(self, model, 
                 model_params, 
                 state_processor, 
                 n_actions,
                 gamma = 0.99,
                 epsilon = 1.0,
                 min_epsilon = 1e-2,
                 epsilon_decay = .999,
                 loss_function = F.smooth_l1_loss, 
                 optimizer = optim.Adam,
                 learning_rate = 1e-3,
                 l2_regularization = 0.0,
                 batch_size = 32,
                 replay_memory = 1000000,
                 alpha = 0.6,
                 beta = 0.4,
                 beta_annealing_episodes = 250,
                 replay_start = 50000,
                 target_update_freq = 1000,
                 action_repeat = 4,
                 update_freq = 4, 
                 random_seed = None):
        
        super().__init__(model, 
                         model_params, 
                         state_processor, 
                         n_actions,
                         gamma = gamma,
                         epsilon = epsilon,
                         min_epsilon = min_epsilon,
                         epsilon_decay = epsilon_decay,
                         loss_function = loss_function, 
                         optimizer = optimizer,
                         learning_rate = learning_rate,
                         l2_regularization = l2_regularization,
                         batch_size = batch_size,
                         replay_memory = replay_memory,
                         replay_start = replay_start,
                         target_update_freq = target_update_freq,
                         action_repeat = action_repeat,
                         update_freq = update_freq, 
                         random_seed = random_seed)
        
        self.alpha = alpha
        self.beta = beta
        if beta_annealing_episodes is None:
            self.beta_growth = 0.0
        else:
            self.beta_growth = (1.0 - self.beta) / beta_annealing_episodes
        self.replay_buffer = PrioritizedReplayBuffer(memory_size = replay_memory, alpha = alpha, seed = random_seed)
    
    def update(self):
        if self.current_step >= self.replay_start:
            if self.current_step % self.target_update_freq == 0:
                self.assign_variables()
                
            if self.current_step % self.update_freq == 0:
                samples = self.replay_buffer.sample(self.batch_size, beta = self.beta)
                experiences, is_weights, indices = zip(*samples)
                state, action, reward, next_state, terminal = zip(*experiences)
                
                state = torch.cat(state)
                action = torch.tensor(action, dtype = torch.int64)
                reward = torch.tensor(reward, dtype = torch.float32)
                next_state = torch.cat(next_state)
                
                is_weights = torch.tensor(is_weights, dtype = torch.float32)
                
                update_target = self.update_target(state, action, reward, next_state, terminal, process_state = False)
                Q_sa = self.online_network(state).gather(1, action.unsqueeze(1)).squeeze()
                
                loss = self.loss_function(Q_sa, update_target, is_weights)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                updated_p = np.abs(update_target.detach().numpy() - Q_sa.detach().numpy()) + 1.0
                self.replay_buffer.update_priorities(indices, updated_p)
            
        self.current_step += 1
    
    def update_beta(self):
        self.beta = min(1.0, self.beta + self.beta_growth)
        
class PrioritizedDoubleDQNAgent(PrioritizedDQNAgent):
    def update_target(self, state, action, reward, next_state, terminal, process_state = True):
        Q_s_next = self.estimate_q(next_state, process_state = process_state)
        max_action = torch.max(Q_s_next, dim = 1)[1]
        
        Q_s_next = self.estimate_target_q(next_state, process_state = process_state)
        Q_max = Q_s_next.gather(1, max_action.unsqueeze(1)).squeeze()

        terminal_mask = torch.tensor([not t for t in terminal], dtype = torch.float32)
        update_target = reward + self.gamma * Q_max * terminal_mask
        
        return update_target