from typing import Sequence

from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from monai.losses.dice import DiceLoss
from losses.loss import BoundaryLoss, FocalLoss

def load_losses(_losses: Sequence[str], num_classes: int):
    losses = []
    for l in _losses:
        if l == "mse":
            losses.append(MSELoss())
        elif l == "ce":
            losses.append(CrossEntropyLoss())
        elif l == "bce":
            losses.append(BCEWithLogitsLoss())
        elif l == "dice":
            losses.append(DiceLoss(sigmoid=True))
        elif l == "focal":
            losses.append(FocalLoss(num_classes))
        elif l == "boundary":
            losses.append(BoundaryLoss(num_classes))
        
        # focal loss
    
    return losses