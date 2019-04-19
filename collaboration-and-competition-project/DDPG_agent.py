import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from replay_buffers import ReplayBuffer
from exploration_noise import OrnsteinUhlenbeckProcess

class DDPGAgent():
    def __init__(self, model_fn,
                 action_scale = 1.0,
                 gamma = 0.99,
                 exploration_noise = None,
                 batch_size = 64,
                 replay_memory = 100000,
                 replay_start = 1000,
                 tau = 1e-3,
                 optimizer = optim.Adam,
                 actor_learning_rate = 1e-4,
                 critic_learning_rate = 1e-3,
                 clip_gradients = None,
                 action_repeat = 1,
                 update_freq = 1,
                 random_seed = None):
        # create online and target networks
        self.online_network = model_fn()
        self.target_network = model_fn()
        
        # create the optimizers for the online_network
        self.actor_optimizer = optimizer(self.online_network.actor_params, 
                                         lr = actor_learning_rate)
        self.critic_optimizer = optimizer(self.online_network.critic_params, 
                                          lr = critic_learning_rate)
        self.clip_gradients = clip_gradients
                                         
        # assign the online network variables to the target network
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        self.replay_buffer = ReplayBuffer(memory_size = replay_memory, seed = random_seed)
        self.exploration_noise = exploration_noise
        
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_start = replay_start
        self.action_repeat = action_repeat
        self.update_freq = update_freq
        self.random_seed = random_seed
        
        self.reset_current_step()
        
    def reset_current_step(self):
        self.current_step = 0
        
    def soft_update(self):
        for target_param, online_param in zip(self.target_network.parameters(), self.online_network.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.tau) + online_param * self.tau)
    
    def add_to_replay_memory(self, state, action, reward, next_state, terminal):
        experience = (state, action, reward, next_state, terminal)
        self.replay_buffer.add(experience)
    
    def action(self, state):
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
        
        if (self.current_step % self.action_repeat == 0) or (not hasattr(self, '_previous_action')):
            action = self.online_network(state)
            action = action.squeeze().detach().numpy()
            
            if self.exploration_noise is not None:
                action = action + self.exploration_noise.sample()
                action = np.clip(action, -self.action_scale, self.action_scale)
        else:
            action = self._previous_action
                
        self._previous_action = action

        return action
    
    def update_target(self, state, action, reward, next_state, terminal):
        next_action = self.target_network(next_state).detach()
        Q_sa_next = self.target_network.critic_value(next_state, next_action).detach()

        update_target = reward.unsqueeze(-1) + self.gamma * Q_sa_next * (1 - terminal).unsqueeze(-1)
        update_target = torch.tensor(update_target, dtype = torch.float32)

        return update_target
    
    def update(self, state, action, reward, next_state, terminal):
        self.add_to_replay_memory(state, action, reward, next_state, terminal)
        
        if terminal and (self.exploration_noise is not None):
            try:
                self.exploration_noise[i].reset_states()
            except:
                pass
        
        if self.current_step >= self.replay_start:
            if self.current_step % self.update_freq == 0:
                experiences = self.replay_buffer.sample(self.batch_size)     
                state, action, reward, next_state, terminal = zip(*experiences)
                
                state = torch.tensor(state, dtype = torch.float32)
                action = torch.tensor(action, dtype = torch.float32)
                reward = torch.tensor(reward, dtype = torch.float32)
                next_state = torch.tensor(next_state, dtype = torch.float32)
                terminal = torch.tensor(terminal, dtype = torch.float32)
                
                update_target = self.update_target(state, action, reward, next_state, terminal)
                Q_sa = self.online_network.critic_value(state, action)
                critic_loss = (Q_sa - update_target).pow(2).mul(0.5).sum(-1).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(self.online_network.critic_params, self.clip_gradients)
                self.critic_optimizer.step()

                action = self.online_network(state)
                policy_loss = -self.online_network.critic_value(state, action).mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(self.online_network.actor_params, self.clip_gradients)
                self.actor_optimizer.step()
                
                self.soft_update()
            
        self.current_step += 1