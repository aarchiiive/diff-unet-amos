import torch
import torch.nn as nn   

from models.swin_unetr import SwinUNETRDenoiser
from models.swin_diff_unetr import SwinDiffUNETR

def test_model():   
    device = torch.device("cpu")

    # x = torch.randn((1, 1, 96, 96, 96))
    t = torch.randn((2, 1)).to(device)
    image = torch.randn((1, 1, 96, 96, 96)).to(device)
    label = torch.randn((1, 13, 96, 96, 96)).to(device)
    embeddings = torch.randn((1, 13, 96, 96, 96)).to(device)

    # model = SwinUNETRDenoiser(img_size=96, in_channels=1, out_channels=13)
    model = SwinDiffUNETR(out_channels=13, feature_size=48).to(device)

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
            transforms.RandScaleCropd(
                keys=["image", "label"], 
                roi_scale=[0.75, 0.85, 1.0],
                random_size=False
            ),
            transforms.Resized(keys=["image", "label"], spatial_size=(96*2, 96*2, 96*2)),
            # transforms.RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(96, 96, 96),
            #     pos=1,
            #     neg=1,
            #     num_samples=1,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            
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
        num_workers=2,
        batch_size=1, 
        shuffle=True
    )

    for d in dataloader:
        print(d["image"].shape)
        print(d["label"].shape)
        break
    
if __name__ == "__main__":
    test_dataset()
    # test_model()