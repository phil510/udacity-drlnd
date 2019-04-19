import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
        
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

        
class MADDPGModel(nn.Module):
    def __init__(self, n_agents, state_dims, action_dims, 
                 critic_hidden_layers = (128, 64), 
                 actor_hidden_layers = (128, 64)):
        super().__init__()
            
        actor_layers_input_size = state_dims
        actor_layers = []
        for units in actor_hidden_layers:
            actor_layers.append(nn.Linear(actor_layers_input_size, units))
            actor_layers_input_size = units
        
        if actor_layers:
            self.actor_layers = nn.ModuleList(actor_layers)
          
        # add the action values to the first layer instead of the 2nd layer, which
        # was used in the original DDPG paper 
        critic_layers_input_size = (state_dims + action_dims) * n_agents
        critic_layers = []
        for units in critic_hidden_layers:
            critic_layers.append(nn.Linear(critic_layers_input_size, units))
            critic_layers_input_size = units
        
        if critic_layers:
            self.critic_layers = nn.ModuleList(critic_layers)
        
        self.actor_output_layer = nn.Linear(actor_layers_input_size, action_dims)
        torch.nn.init.uniform_(self.actor_output_layer.weight, -3e-3, 3e-3)
        self.critic_output_layer = nn.Linear(critic_layers_input_size, 1)
        
        # parameter lists for optimizing the actor and critic networks
        self.actor_params = list(self.actor_layers.parameters()) + list(self.actor_output_layer.parameters())
        self.critic_params = list(self.critic_layers.parameters()) + list(self.critic_output_layer.parameters())
        
    def forward(self, state):
        # applying the tanh activations in forward as they are stateless functions
        x = state
        for layer in self.actor_layers:
            x = F.relu(layer(x))
        
        # action returned is scaled between -1 and 1 from tanh activation
        # the DDPG agent scales the action vector in the action function
        action = F.tanh(self.actor_output_layer(x))
        
        return action
        
    def critic_value(self, state, action):
        x = torch.cat((state, action), dim = 1)
        for layer in self.critic_layers:
            x = F.relu(layer(x))
        
        value = self.critic_output_layer(x)
        
        return value
        
class DDPGModel(nn.Module):
    def __init__(self, state_dims, action_dims, 
                 shared_hidden_layers = tuple(), 
                 critic_hidden_layers = (128, 64), 
                 actor_hidden_layers = (128, 64)):
        super().__init__()
        
        input_size = state_dims
        shared_layers = []
        for units in shared_hidden_layers:
            shared_layers.append(nn.Linear(input_size, units))
            input_size = units
        
        if shared_layers:
            self.shared_layers = nn.ModuleList(shared_layers)
            
        actor_layers_input_size = input_size
        actor_layers = []
        for units in actor_hidden_layers:
            actor_layers.append(nn.Linear(actor_layers_input_size, units))
            actor_layers_input_size = units
        
        if actor_layers:
            self.actor_layers = nn.ModuleList(actor_layers)
          
        # add the action values to the first layer instead of the 2nd layer, which
        # was used in the original DDPG paper 
        critic_layers_input_size = input_size + action_dims
        critic_layers = []
        for units in critic_hidden_layers:
            critic_layers.append(nn.Linear(critic_layers_input_size, units))
            critic_layers_input_size = units
        
        if critic_layers:
            self.critic_layers = nn.ModuleList(critic_layers)
        
        self.actor_output_layer = nn.Linear(actor_layers_input_size, action_dims)
        self.critic_output_layer = nn.Linear(critic_layers_input_size, 1)
        
        # parameter lists for optimizing the actor and critic networks
        if hasattr(self, 'shared_layers'):
            self.shared_params = list(self.shared_layers.parameters())
        else:
            self.shared_params = []
        self.actor_params = list(self.actor_layers.parameters()) + list(self.actor_output_layer.parameters()) + self.shared_params
        self.critic_params = list(self.critic_layers.parameters()) + list(self.critic_output_layer.parameters()) + self.shared_params
        
    def forward(self, state):
        # applying the tanh activations in forward as they are stateless functions
        x = state
        if hasattr(self, 'shared_layers'):
            for layer in self.shared_layers:
                x = F.relu(layer(x))

        for layer in self.actor_layers:
            x = F.relu(layer(x))
        
        # action returned is scaled between -1 and 1 from tanh activation
        # the DDPG agent scales the action vector in the action function
        action = F.tanh(self.actor_output_layer(x))
        
        return action
        
    def critic_value(self, state, action):
        x = torch.cat((state, action), dim = 1)
        if hasattr(self, 'shared_layers'):
            for layer in self.shared_layers:
                x = F.relu(layer(x))

        for layer in self.critic_layers:
            x = F.relu(layer(x))
        
        value = self.critic_output_layer(x)
        
        return value
        
# Mulit-Agent Distributional Deep Deterministic Policy Gradients
class MAD3PGModel(nn.Module):
    def __init__(self, n_agents, state_dims, action_dims, quantiles,
                 critic_hidden_layers = (128, 64), 
                 actor_hidden_layers = (128, 64)):
        super().__init__()
        
        actor_layers_input_size = state_dims
        actor_layers = []
        for units in actor_hidden_layers:
            actor_layers.append(nn.Linear(actor_layers_input_size, units))
            actor_layers_input_size = units
        
        if actor_layers:
            self.actor_layers = nn.ModuleList(actor_layers)
          
        # add the action values to the first layer instead of the 2nd layer, which
        # was used in the original DDPG paper 
        critic_layers_input_size = (state_dims + action_dims) * n_agents
        critic_layers = []
        for units in critic_hidden_layers:
            critic_layers.append(nn.Linear(critic_layers_input_size, units))
            critic_layers_input_size = units
        
        if critic_layers:
            self.critic_layers = nn.ModuleList(critic_layers)
        
        self.actor_output_layer = nn.Linear(actor_layers_input_size, action_dims)
        torch.nn.init.uniform_(self.actor_output_layer.weight, -3e-3, 3e-3)
        self.critic_output_layer = nn.Linear(critic_layers_input_size, quantiles)
        
        # parameter lists for optimizing the actor and critic networks
        self.actor_params = list(self.actor_layers.parameters()) + list(self.actor_output_layer.parameters())
        self.critic_params = list(self.critic_layers.parameters()) + list(self.critic_output_layer.parameters())
        
    def forward(self, state):
        # applying the tanh activations in forward as they are stateless functions
        x = state
        for layer in self.actor_layers:
            x = F.relu(layer(x))
        
        # action returned is scaled between -1 and 1 from tanh activation
        # the DDPG agent scales the action vector in the action function
        action = F.tanh(self.actor_output_layer(x))
        
        return action
        
    def critic_value(self, state, action):
        x = torch.cat((state, action), dim = 1)
        for layer in self.critic_layers:
            x = F.relu(layer(x))
        
        quantiles = self.critic_output_layer(x)
        
        return quantiles