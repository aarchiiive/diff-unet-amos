import os
import wandb
import pickle

import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch 


with open('logs/diff-swin-unetr-btcv-1/results.pkl', 'rb') as f:
    results = pickle.load(f)
    
    
images = results['images']
dices = results['dices']
outputs = results['outputs']
labels = results['labels']  

# 클래스 이름
classes = [
    "background", "spleen", "right kidney", "left kidney", "gall bladder",
    "esophagus", "liver", "stomach", "aorta", "IVC",
    "veins", "pancreas", "right adrenal gland", "left adrenal gland"
]

color_map = {
    0: [0, 0, 0],         # 검은색
    1: [255, 0, 0],       # 빨강
    2: [0, 255, 0],       # 초록
    3: [0, 0, 255],       # 파랑
    4: [255, 255, 0],     # 노랑
    5: [0, 255, 255],     # 청록색
    6: [255, 0, 255],     # 자홍색
    7: [0, 255, 127],     # 밝은 청록색
    8: [128, 128, 0],     # 올리브
    9: [128, 0, 128],     # 보라
    10: [255, 165, 0],    # 주황
    11: [255, 192, 203],  # 핑크
    12: [75, 0, 130],     # 인디고
    13: [0, 128, 0]       # 짙은 초록
}

legend_patches = [mpatches.Patch(color=np.array(color)/255.0, label=classes[i]) for i, color in color_map.items()]

def vis_image(x: torch.Tensor, depth: int, index_rate: float = None):
    if index_rate is not None:
        depth = int(x.shape[3]*index_rate)
    
    x = x[:, :, :, :, depth] * 255
    x = x.squeeze(0).permute(1, 2, 0).to(torch.uint8)
    x = x.cpu().numpy()
    
    return x

def vis_label(x: torch.Tensor, depth: int, index_rate: float = None):
    if index_rate is not None:
        depth = int(x.shape[3]*index_rate)
        
    x = torch.argmax(x, dim=1) 
    x = x[:, :, :, depth]
    x = x.permute(1, 2, 0).squeeze(2).to(torch.uint8)
    x = x.cpu().numpy()
    
    rgb_image = np.zeros(x.shape + (3,), dtype=np.uint8)
    for k in color_map:
        rgb_image[x == k] = color_map[k]
        
    return rgb_image

def plot(image, output, label, path):
    plt.figure(figsize=(16, 16))

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.imshow(label, alpha=0.5)
    plt.axis('off')
    plt.title('Image + Label', fontsize=20)

    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.imshow(output, alpha=0.5)
    plt.axis('off')
    plt.title('Image + Output', fontsize=20)

    ax2.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)

    # plt.show()
    plt.tight_layout()  # 레이아웃을 조정
    plt.savefig(path, bbox_inches='tight')  # 여백을 최소화하고 저장
    plt.close()

save_path = 'logs/diff-swin-unetr-btcv-1/vis'
os.makedirs(save_path, exist_ok=True)

patient = 0

for image, output, label in zip(images, outputs, labels):
    d = image.shape[4]  
    frames = []
    
    for i in range(d):
        # if not os.path.exists(os.path.join(save_path, f'{patient}_{i}.png')):
        rgb_image = vis_image(image, i)
        rgb_output = vis_label(output, i)
        rgb_label = vis_label(label, i)
        
        plot(rgb_image, rgb_output, rgb_label, os.path.join(save_path, f'{patient}_{i}.png'))
    
        # frames.append(cv2.imread(os.path.join(save_path, f'{patient}_{i}.png')))
    
    # video = cv2.VideoWriter(f'logs/diff-swin-unetr-btcv-1/{patient}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, frames[0].shape[:2][::-1])

    # for frame in frames:
        # video.write(frame)
    
    # video.release()
    patient += 1
        