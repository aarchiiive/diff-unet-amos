import torch
import torch.nn as nn

class DistanceLabelSmothing(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        alpha: float = 0.1,
        beta: float = 0.1,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor([alpha])) for _ in range(num_classes)])
        self.beta = nn.ParameterList([nn.Parameter(torch.tensor([beta])) for _ in range(num_classes)])
        
    def forward(self, labels: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        print(labels.shape, distances.shape)
        org = labels
        for i in range(self.num_classes):
            labels[:, i] = self.rational(distances[:, i], i) # [0, 1, 0, 0, 0, 0] [0, 0, 1, 0, 0, 0]
            
        labels = torch.abs(org - labels)
        
        return labels
    
    def rational(self, x: torch.Tensor, i: int) -> torch.Tensor:
        return 1 / (self.beta[i] * x + self.epsilon) * self.alpha[i] # 1 / x * alpha = 0.1
    
    def damped_sine(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) / x
    
