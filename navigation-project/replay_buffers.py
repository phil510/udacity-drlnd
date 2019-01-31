import random
from segment_tree import SumSegmentTree, MinSegmentTree
from sum_tree import SumTree

class ReplayBuffer():
    def __init__(self, memory_size = 1000000, seed = None):
        '''
        Replay buffer from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
        
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
        
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, memory_size = 1000000, alpha = 0.5, seed = None):
        '''
        Prioritized replay buffer from https://arxiv.org/pdf/1511.05952.pdf
        This implementation is based on the OpenAI sumtree implemenation which can be found here
        https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
        
        memory_size: int
            maximum number of experiences to store
            
        alpha: float, [0.0, 1.0]
            hyperparameter that controls the amount of prioritization, with 0.0 being 
            no prioritization (the uniform case)
            
        seed: None or int
            random seed for the replay buffer
        '''
        super().__init__(memory_size = memory_size,
                         seed = seed)
        self.alpha = alpha
        
        it_capacity = 1
        while it_capacity < self._memory_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        
    def add(self, experience):
        '''
        Add an experience to the replay buffer
        
        experience: object, usually a tuple
            the experience to store in the replay buffer
            this implementation does not specify a form for the experience as all that is handled by the DQN agent
        '''
        index = self._next_index
        super().add(experience)
        self._it_sum[index] = self._max_priority ** self.alpha
        self._it_min[index] = self._max_priority ** self.alpha
        
    def _sample_proportional(self, batch_size):
        '''
        Function to use sample from the replay buffer with proportional prioritization
        All code here is from the OpenAI implementation to correctly make use of the
        sum-tree data structure
        
        batch_size: int
            then number of experience to sample
            
        res: list
            list of indices of the experiences sampled from the replay buffer
        '''
        res = []
        p_total = self._it_sum.sum(0, len(self._memory) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            index = self._it_sum.find_prefixsum_idx(mass)
            res.append(index)
        
        return res
    
    def sample(self, batch_size, beta = 1.0):
        '''
        Sample from the replay buffer with proportional prioritization
        
        batch_size: int
            then number of experience to sample
            
        samples: list of 3-tuples
            list of sampled experiences, importance sampling weights for each experience, and
            the indices of the experiences (used to update priorities) in the form
            (experience, is_weights, indices)
        '''
        indices = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._memory)) ** (-beta)
        
        samples = []
        for i in indices:
            p_sample = self._it_sum[i] / self._it_sum.sum()
            is_weight = ((p_sample * len(self._memory)) ** (-beta)) / max_weight
            experience = self._memory[i]
            
            sample = (experience, is_weight, i)
            samples.append(sample)
        
        return samples
    
    def update_priorities(self, indices, priorities):
        '''
        Update the priorities for the experiences corresponding to the given indices
        
        indices: list-like
            list of indices for the experiences/priorities to update
            
        priorities: list-like
            list of new priorities corresponding to the given indices
        '''
        for i, priority in zip(indices, priorities):
            self._it_sum[i] = priority ** self.alpha
            self._it_min[i] = priority ** self.alpha
            self._max_priority = max(self._max_priority, priority)
