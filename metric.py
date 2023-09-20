import torch

def iou_score(preds, labels):
    intersection = torch.sum(preds * labels)
    union = torch.sum((preds + labels) > 0)

    return (intersection / (union + 1e-6)).item()