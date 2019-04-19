import random

class ReplayBuffer():
    def __init__(self, memory_size = 1000000, seed = None):
        '''
        Replay buffer class
        
        memory_size: int
            maximum number of experiences to store
            
        seed: None or int
            random seed for the replay buffer
        '''
        self._memory_size = memory_size
        self._memory = []
        self.seed = random.seed(seed)
        
        self._next_index = 0
        
    def __len__(self):
        return len(self._memory)
        
    def add(self, experience):
        '''
        Add an experience to the replay buffer
        
        experience: object, usually a tuple
            the experience to store in the replay buffer
            this implementation does not specify a form for the experience as all that is handled by the DQN agent
        '''
        if self._next_index >= len(self._memory):
            self._memory.append(experience)
        else:
            self._memory[self._next_index] = experience
        self._next_index = (self._next_index + 1) % self._memory_size
        
    def sample(self, batch_size, **kwargs):
        '''
        Randomly sample from the replay buffer using the uniform distribution
        
        batch_size: int
            then number of experience to sample
            
        experiences: list
            list of sampled experiences
        '''
        experiences = [self._memory[i] for i in [random.randint(0, len(self._memory) - 1) for _ in range(batch_size)]]
        
        return experiences