import numpy as np
import gym
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class Model(nn.Module):
    def __init__(self, state_dims, action_dims, 
                 shared_hidden_layers, 
                 value_hidden_layers, 
                 policy_hidden_layers):
        super().__init__()
        
        input_size = state_dims
        shared_layers = []
        for units in shared_hidden_layers:
            shared_layers.append(nn.Linear(input_size, units))
            input_size = units
        
        if shared_layers:
            self.shared_layers = nn.ModuleList(shared_layers)
        
        value_layers_input_size = input_size
        value_layers = []
        for units in value_hidden_layers:
            value_layers.append(nn.Linear(value_layers_input_size, units))
            value_layers_input_size = units
        
        if value_layers:
            self.value_layers = nn.ModuleList(value_layers)
            
        policy_layers_input_size = input_size
        policy_layers = []
        for units in policy_hidden_layers:
            policy_layers.append(nn.Linear(policy_layers_input_size, units))
            policy_layers_input_size = units
        
        if policy_layers:
            self.policy_layers = nn.ModuleList(policy_layers)
        
        self.mu_output_layer = nn.Linear(policy_layers_input_size, action_dims)
        #self.sigma_output_layer = nn.Linear(policy_layers_input_size, action_dims)
        
        self.critic_output_layer = nn.Linear(value_layers_input_size, 1)
        
        self.std = nn.Parameter(torch.zeros(action_dims))
        
    def forward(self, state, action = None):
        # applying the tanh activations in forward as they are stateless functions
        x = state
        if hasattr(self, 'shared_layers'):
            for layer in self.shared_layers:
                x = F.tanh(layer(x))
        
        value_layer = x
        for layer in self.value_layers:
            value_layer = F.tanh(layer(value_layer))
            
        policy_layer = x
        for layer in self.policy_layers:
            policy_layer = F.tanh(layer(policy_layer))
        
        V = self.critic_output_layer(value_layer)
        
        mu = F.tanh(self.mu_output_layer(policy_layer))
        #sigma = F.softplus(self.sigma_output_layer(policy_layer)) + 1e-10
        sigma = F.softplus(self.std)
        action_distribution = torch.distributions.Normal(mu, sigma)
        
        if action is None:
            action = action_distribution.sample()
            action = action.clamp(-1, 1)
        
        action_log_prob = action_distribution.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = action_distribution.entropy().sum(-1).unsqueeze(-1)
        return {'action_distribution': action_distribution,
                'action': action,
                'action_log_prob': action_log_prob,
                'entropy': entropy,
                'V': V}
                
class PPOAgent():
    def __init__(self, state_dims, action_dims,
                 n_workers = 20,
                 n_steps = 5,
                 n_epochs = 5,
                 n_batches = 1,
                 clip_epsilon = 0.2,
                 gamma = 0.99,
                 value_weight = 1.0,
                 entropy_beta = 1e-2,
                 use_gae = False,
                 gae_lambda = 0.95,
                 normalize_adv = True,
                 shared_hidden_layers = (128, ),
                 value_hidden_layers = (128, ),
                 policy_hidden_layers = (128, ),
                 optimizer = optim.Adam,
                 learning_rate = 1e-3,
                 l2_regularization = 0.0,
                 gradient_clipping = None):
        
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.model = Model(state_dims, action_dims, 
                           shared_hidden_layers, 
                           value_hidden_layers,
                           policy_hidden_layers)
        
        self.n_workers = n_workers
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.value_weight = value_weight
        self.entropy_beta = entropy_beta
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        
        self.optimizer = optimizer(self.model.parameters(), 
                                   lr = learning_rate,
                                   weight_decay = l2_regularization)
        self.gradient_clipping = gradient_clipping
        
        self.reset_current_step()
        self.reset_histories()
        
    def reset_current_step(self):
        self.current_step = 0
        
    def reset_histories(self):
        self.value_history = deque(maxlen = self.n_steps)
        self.experience_history = deque(maxlen = self.n_steps)
        self.act_log_prob_history = deque(maxlen = self.n_steps)
        
    def action(self, state):
        state = torch.tensor(state, dtype = torch.float32)
        prediction = self.model(state)
        
        action = prediction['action'].detach().numpy()
        action_log_prob = prediction['action_log_prob'].detach().numpy().squeeze()
        value = prediction['V'].detach().numpy().squeeze()

        self.value_history.append(value)
        self.act_log_prob_history.append(action_log_prob)
        
        return action
        
    def update_target(self, state, action, reward, next_state, terminal):
        returns = []
        advantage = []
        
        next_V = self.model(torch.tensor(next_state[-1], dtype = torch.float32))['V'].detach().numpy().squeeze()
        R = next_V
        A = np.zeros(self.n_workers)
        for r, d, V in zip(reversed(reward), reversed(terminal), reversed(self.value_history)):          
            R = r + self.gamma * R * (1 - d) # (1 - done) is used to "start over" when there is a terminal state
            returns.append(R)
            
            if self.use_gae:
                td_error = r + self.gamma * next_V * (1 - d) - V
                A = A * self.gae_lambda * self.gamma * (1 - d) + td_error
                next_V = V
                advantage.append(A)
            else:
                A = R - V
                advantage.append(A)

        returns = returns[::-1]
        advantage = advantage[::-1]
        
        batch_shape = ((self.n_steps * self.n_workers), 1)
        returns = np.asarray(returns).swapaxes(1, 0).reshape(batch_shape)
        advantage = np.asarray(advantage).swapaxes(1, 0).reshape(batch_shape)
        
        if self.normalize_adv:
            advantage = (advantage - np.mean(advantage)) / np.std(advantage)

        return returns, advantage
        
    def update(self, state, action, reward, next_state, terminal):
        self.experience_history.append((state, action, reward, next_state, terminal))
        
        if len(self.experience_history) == self.n_steps:
            state, action, reward, next_state, terminal = zip(*self.experience_history)

            returns, advantage = self.update_target(state, action, reward, next_state, terminal)
            
            # reshape everything so that they are (batch_size, variable_shape) where batch_size = n_workers * n_steps
            state_batch_shape = ((self.n_steps * self.n_workers), ) + (self.state_dims, )
            action_batch_shape = ((self.n_steps * self.n_workers), ) + (self.action_dims, )
            batch_shape = ((self.n_steps * self.n_workers), 1)
            
            state = np.asarray(state).swapaxes(1, 0).reshape(state_batch_shape)
            action = np.asarray(action).swapaxes(1, 0).reshape(action_batch_shape)
            action_log_probs_old = np.asarray(self.act_log_prob_history).swapaxes(1, 0).reshape(batch_shape)
            
            for _ in range(self.n_epochs):
                indices = np.arange((self.n_steps * self.n_workers))
                np.random.shuffle(indices)
                indices = np.array_split(indices, self.n_batches)
                
                for i in indices:
                    # create mini-batch for optimization
                    state_sample = torch.tensor(state[i], dtype = torch.float32)
                    action_sample = torch.tensor(action[i], dtype = torch.float32)
                    returns_sample = torch.tensor(returns[i], dtype = torch.float32)
                    advantage_sample = torch.tensor(advantage[i], dtype = torch.float32)
                    action_log_probs_old_sample = torch.tensor(action_log_probs_old[i], dtype = torch.float32)

                    prediction = self.model(state_sample, action = action_sample)

                    value_loss = 0.5 * (returns_sample - prediction['V']).pow(2).mean()
                    
                    ratio = (prediction['action_log_prob'] - action_log_probs_old_sample.detach()).exp()
                    policy_loss = -torch.min(ratio * advantage_sample.detach(), ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_sample.detach()).mean()
                                           
                    entropy_regularization = prediction['action_distribution'].entropy().sum(-1).mean()
                    
                    loss = (value_loss * self.value_weight) + policy_loss - (self.entropy_beta * entropy_regularization)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.gradient_clipping:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    self.optimizer.step()
            
            self.reset_histories()
        
        self.current_step += self.n_workers