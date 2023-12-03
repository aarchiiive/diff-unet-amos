import time

import torch
import torch.nn as nn
import torch.nn.functional as F   


class LabelSmoothing:
    """Smoothing labels based on distanced between labels and one-hot labels."""
    def __init__(self, num_classes: int, epsilon: float = 0.0):
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.confidence = 1.0 - epsilon
    
    
    # def __call__(self, labels: torch.Tensor) -> torch.Tensor:
    #     """Callable function for smoothing labels.

    #     Args:
    #         labels (torch.Tensor): One-hot encoded label -> (B, C, D, H, W) 

    #     Returns:
    #         labels (torch.Tensor): Smoothed label(similar to one-hot format) -> (B, C, D, H, W) 
    #         Example : [0, 0, 0, 1, 0, 0, 0] -> [0.05, 0.05, 0.05, 0.8, 0.05, 0.05, 0.05]
    #     """
    #     B, C, D, H, W = labels.shape  # [B, C, D, H, W]
    #     reduced_labels = torch.argmax(labels, dim=1, keepdim=True)

    #     # Calculate class-wise coordinates
    #     class_coordinates = [torch.stack(torch.where(reduced_labels == i)[2:], dim=1).float() for i in range(self.num_classes)] # [[N1, 3], [N2, 3], ...]

    #     # Calculate class-wise centroids
    #     class_centroids = [torch.mean(coords, dim=0, dtype=labels.dtype).to(labels.device) for coords in class_coordinates] # [[3], [3], ...]
        
        # for x in range(W):
        #     for y in range(H):
        #         for z in range(D):
        #             for n in range(C):
        #                 labels[:, n, z, y, x] = torch.norm(torch.tensor([x, y, z]).to(labels.device) - class_centroids[n], dim=-1)
        #                 # print(labels[:, n, z, y, x])

    #     # weights = distances / distances.max()
        
    #     # print(distances.shape)
    #     # print(distances[:, :, 12, 12, 12])
    #     # print(weights[:, :, 12, 12, 12])
    #     # print(labels[:, :, 12, 12, 12])

    #     return labels
    
    def __call__(self, labels: torch.Tensor) -> torch.Tensor:
        """Callable function for smoothing labels.

        Args:
            labels (torch.Tensor): One-hot encoded label -> (B, C, D, H, W) 

        Returns:
            labels (torch.Tensor): Smoothed label (similar to one-hot format) -> (B, C, D, H, W) 
            Example: [0, 0, 0, 1, 0, 0, 0] -> [0.05, 0.05, 0.05, 0.8, 0.05, 0.05, 0.05]
        """
        B, C, D, H, W = labels.shape

        reduced_labels = torch.argmax(labels, dim=1, keepdim=True)
        
        # Calculate class-wise coordinates
        class_coordinates = [torch.stack(torch.where(reduced_labels == i)[2:], dim=1).float() for i in range(self.num_classes)] # [[N1, 3], [N2, 3], ...]

        # Calculate class-wise centroids
        class_centroids = [torch.mean(coords, dim=0, dtype=labels.dtype).to(labels.device) for coords in class_coordinates] # [[3], [3], ...]

        # Calculate class-wise coordinates
        for i in range(self.num_classes):
            for t in torch.nonzero(reduced_labels == i, as_tuple=True): # [B, 1, D, H, W]
                # print(t)
                print(t.shape)
                print(t)
            break
        
        for x in range(W):
            for y in range(H):
                for z in range(D):
                    for n in range(C):
                        labels[:, n, z, y, x] = torch.norm(torch.tensor([x, y, z]).to(labels.device) - class_centroids[n], dim=-1)
                        print(labels[:, n, ...])

            # print(torch.nonzero(reduced_labels == i, as_tuple=True))
        # class_coordinates = [torch.nonzero(reduced_labels == i, as_tuple=True)[2:].float().t() for i in range(self.num_classes)]

        # # Calculate class-wise centroids
        # class_centroids = [coords.mean(dim=0).to(labels.device) for coords in class_coordinates]

        # # Calculate distances and update labels
        # grid_coords = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), torch.arange(D)), dim=-1).to(labels.device)
        # print(grid_coords.shape)
        # distances = torch.norm(grid_coords[:, None, None, None, None, :] - class_centroids, dim=-1)
        # print(distances.shape)
        # labels = distances.permute(0, 3, 2, 1, 4)

        return labels

if __name__ == "__main__":
    # test code
    device = torch.device("cpu")
    label = torch.randint(0, 13, (2, 96, 96, 96)).to(device)
    label = F.one_hot(label, num_classes=13).permute(0, 4, 1, 2, 3).float()
    t0 = time.time()
    label = LabelSmoothing(num_classes=13, epsilon=0.1)(label)
    print(f"Time : {time.time() - t0:.4f}s")