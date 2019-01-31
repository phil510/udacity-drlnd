import torch

def weighted_mse_loss(input, target, weight):
    loss = ((input - target) ** 2) * weight
    loss = torch.mean(loss)
    
    return loss
    
def weighted_smooth_l1_loss(input, target, weight):
    loss = torch.abs(input - target)
    loss = torch.where(loss < 1, (0.5 * loss ** 2) * weight, (loss - 0.5) * weight)
    loss = torch.mean(loss)
    
    return loss