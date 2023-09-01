import torch
import numpy as np

def dice_coef(result, reference):
    r"""
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.
    
    The metric is defined as
    
    .. math::
        
        DC=\frac{2|A\cap B|}{|A|+|B|}
        
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    
    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def dice_score(pred, target, class_idx):         
    smooth = 1e-5  # Smoothing factor to avoid division by zero

    pred_class = (pred == class_idx).float()
    target_class = (target == class_idx).float()

    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice

def hd95_score(pred, target, class_idx):
    pred_points = torch.argwhere(pred == class_idx)
    target_points = torch.argwhere(target == class_idx)

    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')

    hausdorff_distances = []
    for target_point in target_points:
        distances = torch.norm(pred_points - target_point, dim=1)
        distance = torch.min(distances)
        hausdorff_distances.append(distance.item())

    percentile_95 = torch.tensor(hausdorff_distances).percentile(95).item()
    return percentile_95
