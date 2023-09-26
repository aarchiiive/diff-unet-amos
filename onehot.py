import torch
from torch.nn.functional import one_hot

num_classes = 6
labels = torch.randint(0, 5, (4, 1, 512 ,512))
labels = torch.cat([labels == i for i in range(num_classes)], dim=1)
print(labels.shape)