import time

import torch
import torch.nn as nn   

from medpy.metric.binary import hd, dc, assd

from metric import dice_coeff

from models.swin_unetr import SwinUNETRDenoiser
from models.diff_swin_unetr import DiffSwinUNETR

def test_model():   
    device = torch.device("cpu")

    # x = torch.randn((1, 1, 96, 96, 96))
    t = torch.randn((2, 1)).to(device)
    image = torch.randn((1, 1, 96, 96, 96)).to(device)
    label = torch.randn((1, 13, 96, 96, 96)).to(device)
    embeddings = torch.randn((1, 13, 96, 96, 96)).to(device)

    # model = SwinUNETRDenoiser(img_size=96, in_channels=1, out_channels=13)
    model = DiffSwinUNETR(out_channels=13, feature_size=48).to(device)

    model.embed_model.swinViT.load_state_dict(torch.load("pretrained/swin_unetr/swinvit.pt"))

    with torch.no_grad():   
        x_start = (label) * 2 - 1
        x_t, t, _ = model(x=x_start, pred_type="q_sample")
        pred = model(x=x_t, step=t, image=image, pred_type="denoise")

    # out = model(x, t, embeddings)
    
def test_dataset():
    import os

    from monai import transforms
    from monai.data import (
        ThreadDataLoader,
        # CacheDataset,
        load_decathlon_datalist,
        decollate_batch,
        set_track_meta,
    )

    from dataset.cache_dataset import LabelSmoothingCacheDataset

    data_path = "../datasets/BTCV/"
    num_classes = 14

    transform = transforms.Compose(
        [   
            # transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image"
            ),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            # transforms.RandScaleCropd(
            #     keys=["image", "label"], 
            #     roi_scale=[0.75, 0.85, 1.0],
            #     random_size=False
            # ),
            # transforms.Resized(keys=["image", "label"], spatial_size=(96*2, 96*2, 96*2)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            
            transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),

            # transforms.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    data = load_decathlon_datalist(os.path.join(data_path, "dataset.json"), True, "validation")
    dataset = LabelSmoothingCacheDataset(
        data=data,
        transform=transform,
        cache_num=len(data),
        cache_rate=1.0,
        num_workers=max(2, 20),
        num_classes=num_classes,
    )
    dataloader = ThreadDataLoader(
        dataset=dataset,
        num_workers=1,
        batch_size=1, 
        shuffle=True
    )

    for d in dataloader:
        print(d["image"].shape)
        print(d["label"].shape)
        # break
    
    


def dice_coeff(result: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
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
    result : torch.Tensor
        Input data containing objects. Should be a torch tensor with dtype torch.bool,
        where 0 represents background and 1 represents the object.
    reference : torch.Tensor
        Input data containing objects. Should be a torch tensor with dtype torch.bool,
        where 0 represents background and 1 represents the object.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    if isinstance(result, torch.FloatTensor) or result.dtype == torch.float:
        intersection = torch.sum(result.bool() & reference.bool()).item()
    elif isinstance(result, torch.LongTensor) or result.dtype == torch.int:
        intersection = torch.sum(result & reference).item()
    
    size_i1 = torch.sum(result).item()
    size_i2 = torch.sum(reference).item()
    
    try:
        dc_value = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc_value = 0.0
    
    return torch.tensor(dc_value)
    
def test_dice():
    t0 = time.time()
    
    device = torch.device("cuda:0")
    
    dices = []
    num_classes = 13
    outputs = torch.randint(0, 2, (1, 13, 96, 96, 96)).float().to(device)
    labels = torch.randint(0, 2, (1, 13, 96, 96, 96)).float().to(device)
    
    for i in range(num_classes):
        output = outputs[:, i]
        label = labels[:, i]
        if output.sum() > 0 and label.sum() == 0:
            dice = torch.Tensor([1.0]).to(outputs.device)
        else:
            dice = dice_coeff(output, label).to(outputs.device)
            
        dices.append(dice)
        dices.append(torch.Tensor([1.0]).squeeze().to(outputs.device))
        
    
    print(dices)
    
    dices = torch.stack(dices)
    
    print(dices)
    print(f"Time: {time.time() - t0:.4f}s")
    
if __name__ == "__main__":
    test_dice()
    # test_dataset()
    # test_model()