import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from replay_buffers import ReplayBuffer
from exploration_noise import OrnsteinUhlenbeckProcess

class MADDPGAgent():
    def __init__(self, n_agents, model_fn,
                 action_scale = 1.0,
                 gamma = 0.99,
                 exploration_noise_fn = None,
                 batch_size = 64,
                 replay_memory = 100000,
                 replay_start = 100,
                 tau = 1e-3,
                 optimizer = optim.Adam,
                 actor_learning_rate = 1e-4,
                 critic_learning_rate = 1e-3,
                 clip_gradients = None,
                 share_weights = False,
                 action_repeat = 1,
                 update_freq = 1,
                 random_seed = None):
        # create online and target networks for each agent
        self.n_agents = n_agents
        
        self.online_networks = [model_fn() for _ in range(self.n_agents)]
        self.target_networks = [model_fn() for _ in range(self.n_agents)]
        
        self.actor_optimizers = [optimizer(agent.actor_params, 
                                           lr = actor_learning_rate) for agent in self.online_networks]
        self.critic_optimizers = [optimizer(agent.critic_params, 
                                            lr = critic_learning_rate) for agent in self.online_networks]
        
        if exploration_noise_fn:
            self.exploration_noise = [exploration_noise_fn() for _ in range(self.n_agents)]
        else:
            self.exploration_noise = None
                                         
        # assign the online network variables to the target network
        for target_network, online_network in zip(self.target_networks, self.online_networks):
            target_network.load_state_dict(online_network.state_dict())
        
        self.replay_buffer = ReplayBuffer(memory_size = replay_memory, seed = random_seed)
        
        self.share_weights = share_weights
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.clip_gradients = clip_gradients
        self.replay_start = replay_start
        self.action_repeat = action_repeat
        self.update_freq = update_freq
        self.random_seed = random_seed
        
        self.reset_current_step()
        
    def reset_current_step(self):
        self.current_step = 0
        
    def soft_update(self):
        for target_network, online_network in zip(self.target_networks, self.online_networks):
            for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
                target_param.detach_()
                target_param.copy_(target_param * (1.0 - self.tau) + online_param * self.tau)
                
    def assign_weights(self):
        for target_network, online_network in zip(self.target_networks, self.online_networks):
            target_network.load_state_dict(self.target_networks[0].state_dict())
            online_network.load_state_dict(self.online_networks[0].state_dict())
    
    def add_to_replay_memory(self, state, action, reward, next_state, terminal):
        experience = (state, action, reward, next_state, terminal)
        self.replay_buffer.add(experience)
    
    def action(self, state):
        if (self.current_step % self.action_repeat == 0) or (not hasattr(self, '_previous_action')):
            actions = []
            for i in range(self.n_agents):
                obs = torch.tensor(state[i], dtype = torch.float32).unsqueeze(0)
                action = self.online_networks[i](obs)
                action = action.squeeze().detach().numpy()
                
                if self.exploration_noise:
                    action = action + self.exploration_noise[i].sample()
                    action = np.clip(action, -self.action_scale, self.action_scale)
                actions.append(action)
                
            action = np.asarray(actions)
        else:
            action = self._previous_action
                
        self._previous_action = action

        return action
    
    def update_target(self, state, action, reward, next_state, terminal):
        all_next_actions = []
        for i in range(self.n_agents):
            next_action = self.target_networks[i](next_state[:, i, :])
            all_next_actions.append(next_action)

        all_next_actions = torch.cat(all_next_actions, dim = 1)
        all_next_states = next_state.view(-1, next_state.shape[1] * next_state.shape[2])

        Q_sa_next = self.target_networks[self.current_agent].critic_value(all_next_states, all_next_actions)

        reward = reward[:, self.current_agent].unsqueeze(-1)
        terminal = terminal[:, self.current_agent].unsqueeze(-1)
 
        update_target = reward + self.gamma * Q_sa_next * (1 - terminal)
        update_target = update_target.detach()
        
        return update_target
    
    def update(self, state, action, reward, next_state, terminal):
        self.add_to_replay_memory(state, action, reward, next_state, terminal)
        
        if np.any(terminal) and (self.exploration_noise is not None):
            for i in range(self.n_agents):
                try:
                    self.exploration_noise[i].reset_states()
                except:
                    pass
        
        if self.current_step >= self.replay_start:
            if self.current_step % self.update_freq == 0:
                if self.share_weights:
                    update_agents = 1
                else:
                    update_agents = self.n_agents

                for i in range(update_agents):
                    self.current_agent = i
                
                    experiences = self.replay_buffer.sample(self.batch_size)     
                    state, action, reward, next_state, terminal = zip(*experiences)
                    
                    state = torch.tensor(state, dtype = torch.float32)
                    action = torch.tensor(action, dtype = torch.float32)
                    reward = torch.tensor(reward, dtype = torch.float32)
                    next_state = torch.tensor(next_state, dtype = torch.float32)
                    terminal = torch.tensor(terminal, dtype = torch.float32)

                    all_actions = action.view(-1, action.shape[1] * action.shape[2])
                    all_states = state.view(-1, state.shape[1] * state.shape[2])
                    
                    update_target = self.update_target(state, action, reward, next_state, terminal)
                    
                    Q_sa = self.online_networks[self.current_agent].critic_value(all_states, all_actions)
                    critic_loss = F.mse_loss(Q_sa, update_target)
                    
                    self.critic_optimizers[self.current_agent].zero_grad()
                    critic_loss.backward()
                    if self.clip_gradients:
                        nn.utils.clip_grad_norm_(self.online_networks[self.current_agent].critic_params, self.clip_gradients)
                    self.critic_optimizers[self.current_agent].step()

                    agent_action = self.online_networks[self.current_agent](state[:, self.current_agent, :])

                    predicted_actions = action.clone().detach()
                    predicted_actions[:, self.current_agent] = agent_action
                    predicted_actions = predicted_actions.view(-1, predicted_actions.shape[1] * predicted_actions.shape[2])

                    policy_loss = -self.online_networks[self.current_agent].critic_value(all_states, predicted_actions).mean()

                    self.actor_optimizers[self.current_agent].zero_grad()
                    policy_loss.backward()
                    if self.clip_gradients:
                        nn.utils.clip_grad_norm_(self.online_networks[self.current_agent].actor_params, self.clip_gradients)
                    self.actor_optimizers[self.current_agent].step()
                    
                self.soft_update()
                
                if self.share_weights:
                    self.assign_weights()
            
        self.current_step += 1

def qr_huber_loss(difference, k = 1.0):
    return torch.where(difference.abs() < k, 0.5 * difference.pow(2), k * (difference.abs() - 0.5 * k))
        
class MAD3PGAgent():
    def __init__(self, n_agents, model_fn, n_quantiles,
                 action_scale = 1.0,
                 gamma = 0.99,
                 exploration_noise_fn = None,
                 batch_size = 64,
                 replay_memory = 100000,
                 replay_start = 100,
                 tau = 1e-3,
                 optimizer = optim.Adam,
                 actor_learning_rate = 1e-4,
                 critic_learning_rate = 1e-3,
                 clip_gradients = None,
                 share_weights = False,
                 action_repeat = 1,
                 update_freq = 1,
                 random_seed = None):
        # create online and target networks for each agent
        self.n_agents = n_agents
        
        self.online_networks = [model_fn() for _ in range(self.n_agents)]
        self.target_networks = [model_fn() for _ in range(self.n_agents)]
        
        self.actor_optimizers = [optimizer(agent.actor_params, 
                                           lr = actor_learning_rate) for agent in self.online_networks]
        self.critic_optimizers = [optimizer(agent.critic_params, 
                                            lr = critic_learning_rate) for agent in self.online_networks]
        
        if exploration_noise_fn:
            self.exploration_noise = [exploration_noise_fn() for _ in range(self.n_agents)]
        else:
            self.exploration_noise = None
            
        self.n_quantiles = n_quantiles
        self.cumulative_density = torch.tensor((2 * np.arange(self.n_quantiles) + 1) / (2.0 * self.n_quantiles), 
                                               dtype = torch.float32).view(1, -1)
                                         
        # assign the online network variables to the target network
        for target_network, online_network in zip(self.target_networks, self.online_networks):
            target_network.load_state_dict(online_network.state_dict())
        
        self.replay_buffer = ReplayBuffer(memory_size = replay_memory, seed = random_seed)
        
        self.share_weights = share_weights
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.clip_gradients = clip_gradients
        self.replay_start = replay_start
        self.action_repeat = action_repeat
        self.update_freq = update_freq
        self.random_seed = random_seed
        
        self.reset_current_step()
        
    def reset_current_step(self):
        self.current_step = 0
        
    def soft_update(self):
        for target_network, online_network in zip(self.target_networks, self.online_networks):
            for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
                target_param.detach_()
                target_param.copy_(target_param * (1.0 - self.tau) + online_param * self.tau)
                
    def assign_weights(self):
        for target_network, online_network in zip(self.target_networks, self.online_networks):
            target_network.load_state_dict(self.target_networks[0].state_dict())
            online_network.load_state_dict(self.online_networks[0].state_dict())
    
    def add_to_replay_memory(self, state, action, reward, next_state, terminal):
        experience = (state, action, reward, next_state, terminal)
        self.replay_buffer.add(experience)
    
    def action(self, state):
        if (self.current_step % self.action_repeat == 0) or (not hasattr(self, '_previous_action')):
            actions = []
            for i in range(self.n_agents):
                obs = torch.tensor(state[i], dtype = torch.float32).unsqueeze(0)
                action = self.online_networks[i](obs)
                action = action.squeeze().detach().numpy()
                
                if self.exploration_noise:
                    action = action + self.exploration_noise[i].sample()
                    action = np.clip(action, -self.action_scale, self.action_scale)
                actions.append(action)
                
            action = np.asarray(actions)
        else:
            action = self._previous_action
                
        self._previous_action = action

        return action
    
    def update_target(self, state, action, reward, next_state, terminal):
        all_next_actions = []
        for i in range(self.n_agents):
            next_action = self.target_networks[i](next_state[:, i, :])
            all_next_actions.append(next_action)

        all_next_actions = torch.cat(all_next_actions, dim = 1)
        all_next_states = next_state.view(-1, next_state.shape[1] * next_state.shape[2])

        quantiles_next = self.target_networks[self.current_agent].critic_value(all_next_states, all_next_actions)

        reward = reward[:, self.current_agent].unsqueeze(-1)
        terminal = terminal[:, self.current_agent].unsqueeze(-1)
 
        update_target = reward + self.gamma * quantiles_next * (1 - terminal)
        update_target = update_target.detach()
        
        return update_target
    
    def update(self, state, action, reward, next_state, terminal):
        self.add_to_replay_memory(state, action, reward, next_state, terminal)
        
        if np.any(terminal) and (self.exploration_noise is not None):
            for i in range(self.n_agents):
                try:
                    self.exploration_noise[i].reset_states()
                except:
                    pass
        
        if self.current_step >= self.replay_start:
            if self.current_step % self.update_freq == 0:
                if self.share_weights:
                    update_agents = 1
                else:
                    update_agents = self.n_agents

                for i in range(update_agents):
                    self.current_agent = i
                
                    experiences = self.replay_buffer.sample(self.batch_size)     
                    state, action, reward, next_state, terminal = zip(*experiences)
                    
                    state = torch.tensor(state, dtype = torch.float32)
                    action = torch.tensor(action, dtype = torch.float32)
                    reward = torch.tensor(reward, dtype = torch.float32)
                    next_state = torch.tensor(next_state, dtype = torch.float32)
                    terminal = torch.tensor(terminal, dtype = torch.float32)

                    all_actions = action.view(-1, action.shape[1] * action.shape[2])
                    all_states = state.view(-1, state.shape[1] * state.shape[2])
                    
                    update_target = self.update_target(state, action, reward, next_state, terminal)
                    
                    quantiles = self.online_networks[self.current_agent].critic_value(all_states, all_actions)
                    
                    update_target = update_target.t().unsqueeze(-1)
                    difference = update_target - quantiles
                    critic_loss = qr_huber_loss(difference) * (self.cumulative_density - (difference.detach() < 0).float()).abs()
                    critic_loss = critic_loss.mean(0).mean(1).sum()
                    
                    self.critic_optimizers[self.current_agent].zero_grad()
                    critic_loss.backward()
                    if self.clip_gradients:
                        nn.utils.clip_grad_norm_(self.online_networks[self.current_agent].critic_params, self.clip_gradients)
                    self.critic_optimizers[self.current_agent].step()

                    agent_action = self.online_networks[self.current_agent](state[:, self.current_agent, :])

                    predicted_actions = action.clone().detach()
                    predicted_actions[:, self.current_agent] = agent_action
                    predicted_actions = predicted_actions.view(-1, predicted_actions.shape[1] * predicted_actions.shape[2])

                    policy_loss = -self.online_networks[self.current_agent].critic_value(all_states, predicted_actions).mean()

                    self.actor_optimizers[self.current_agent].zero_grad()
                    policy_loss.backward()
                    if self.clip_gradients:
                        nn.utils.clip_grad_norm_(self.online_networks[self.current_agent].actor_params, self.clip_gradients)
                    self.actor_optimizers[self.current_agent].step()
                    
                self.soft_update()
                
                if self.share_weights:
                    self.assign_weights()
            
        self.current_step += 1
        