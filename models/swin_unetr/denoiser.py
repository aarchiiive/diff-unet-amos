# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import math
import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import save_image

from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from .vxm import VXM
from .layers import SmoothLayer
from .patch_embed import PatchEmbed
from .blocks import UnetOutBlock, UnetrUpBlock, UnetrBasicBlock
from .patch import MERGING_MODE
from .transformer import SwinTransformer
from ..diffusion import get_timestep_embedding, nonlinearity, TimeStepEmbedder

rearrange, _ = optional_import("einops", name="rearrange")

class SwinUNETRDenoiser(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        image_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        embedding_size: int = 512,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        embedding_dim: int = 128, # for time embedding
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        mask_ratio: float = None,
        noise_ratio: float = 0.5,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        """
        Args:
            image_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(image_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(image_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(image_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        image_size = ensure_tuple_rep(image_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        
        self.image_size = image_size    
        self.in_channels = in_channels  
        self.num_classes = out_channels

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(image_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (image_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        
        # timesteps & noise
        self.noise_ratio = noise_ratio
        self.t_embedder = TimeStepEmbedder(embedding_dim)
        
        # features = [feature_size * (2 ** max(0, i - 1)) for i in range(4)]
        # self.smooth = nn.ModuleList([
        #     SmoothLayer(features[i], image_size[0] // (2**i), image_size[1] // (2**i), image_size[2] // (2**i)) \
        #         for i in range(len(depths))
        # ])
        
        # self.comp_mixer = CompositionalMixer(in_channels, 2*in_channels, out_channels, image_size)
        
        # self.vxm = VXM(image_size, out_channels, len(image_size))
            
        # voxelmorph
        # self.vxm = None
        # self.spatial_transformer = SpatialTransformer(image_size, mode="bilinear", padding_mode="border")
        # if isinstance(image_size, int):
        #     self.pos_embed = nn.Parameter(torch.zeros(1, in_channels, image_size, image_size, image_size))
        # elif isinstance(image_size, Sequence):
        #     self.pos_embed = nn.Parameter(torch.zeros(1, in_channels, *image_size))

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            embedding_size=embedding_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            embedding_size=embedding_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
    
    def _forward(self, x: torch.Tensor, t, embeddings=None, image=None, label = None):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        
        
        
        if image is not None :
            x = torch.cat([image, x], dim=1) # x = [1,14,96,96,96]

        x0 = self.conv_0(x, temb) # torch.Size([1, 64, 96, 96, 96])
        # c0 = self.conv_0(canny, temb)   # experiment2
        if embeddings is not None:
            x0 += embeddings[0]
        crop_0 = -1*(torch.sigmoid(x0)) + 1
        r0 = torch.mul(x0, crop_0)

        x1 = self.down_1(x0, temb) # torch.Size([1, 64, 48, 48, 48])
        if embeddings is not None:
            x1 += embeddings[1]
        crop_1 = -1*(torch.sigmoid(x1)) + 1
        r1 = torch.mul(x1, crop_1)

        x2 = self.down_2(x1, temb) # torch.Size([1, 128, 24, 24, 24])
        if embeddings is not None:
            x2 += embeddings[2]
        crop_2 = -1*(torch.sigmoid(x2)) + 1
        r2 = torch.mul(x2, crop_2)

        x3 = self.down_3(x2, temb) # torch.Size([1, 256, 12, 12, 12])
        if embeddings is not None:
            x3 += embeddings[3]
        crop_3 = -1*(torch.sigmoid(x3)) + 1
        r3 = torch.mul(x3, crop_3)

        x4 = self.down_4(x3, temb) # torch.Size([1, 512, 6, 6, 6])
        if embeddings is not None:
            x4 += embeddings[4]
        crop_4 = -1*(torch.sigmoid(x4)) + 1
        r4 = torch.mul(x4, crop_4)
        
        u4 = self.upcat_4(x4, x3, temb) # torch.Size([1, 256, 12, 12, 12])
        u3 = self.upcat_3(u4, x2, temb) # torch.Size([1, 128, 24, 24, 24])
        u2 = self.upcat_2(u3, x1, temb) # torch.Size([1, 64, 48, 48, 48])
        u1 = self.upcat_1(u2, x0, temb) # torch.Size([1, 64, 96, 96, 96])

        logits = self.final_conv(u1)  # torch.Size([1, 13, 96, 96, 96])
        
        save_image(x0[0,1,:,:,2], 'pred.png')
        save_image(crop_0[0,1,:,:,2], 'RA.png')
        save_image(r0[0,1,:,:,2], 'output.png')

        # ## Reverse Attention branch
        x = self.conv0(logits)
        x = x + r4
        x = self.upsample0(x)
        x = x + r3
        x = self.upsample1(x)
        x = x + r2
        x = self.upsample2(x)
        x = x + r1
        x = self.upsample3(x)
        x = x + r0
        logits = self.final_conv1(x)

        return logits
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        image: torch.Tensor = None,
        embeddings: Any = None, # possible to include list of tensors
    ):
        t = self.t_embedder(t)
        # noise = x
        x = torch.cat([image, x], dim=1) # + self.pos_embed
        # comp = self.comp_mixer(x) # mixed composition
        
        hidden_states_out = self.swinViT(x, t, self.normalize)
        
        for i in range(len(hidden_states_out)):
            hidden_states_out[i] = hidden_states_out[i] + embeddings[0][i]
        
        enc0 = self.encoder1(x, t) + embeddings[1] # [1, 48, 96, 96, 96]
        r0 = self.reverse_attention(enc0)
        # print("r0:", r0.shape)
        # enc0 = self.smooth[0](enc0)
        enc1 = self.encoder2(hidden_states_out[0], t) + embeddings[2] # [1, 48, 48, 48, 48]
        r1 = self.reverse_attention(enc1)
        # print("r1:", r1.shape)
        # enc1 = self.smooth[1](enc1)
        enc2 = self.encoder3(hidden_states_out[1], t) + embeddings[3] # [1, 96, 24, 24, 24]
        r2 = self.reverse_attention(enc2)
        # print("r2:", r2.shape)
        # enc2 = self.smooth[2](enc2)
        enc3 = self.encoder4(hidden_states_out[2], t) + embeddings[4] # [1, 192, 12, 12, 12]
        r3 = self.reverse_attention(enc3)
        # print("r3:", r3.shape)
        # enc3 = self.smooth[3](enc3)
        
        dec4 = self.encoder10(hidden_states_out[4], t)
        dec3 = self.decoder5(dec4, hidden_states_out[3], t)
        dec2 = self.decoder4(dec3, enc3, t) # [1, 192, 12, 12, 12]
        dec2 = dec2 + r3
        dec1 = self.decoder3(dec2, enc2, t) # [1, 96, 24, 24, 24]
        dec1 = dec1 + r2
        dec0 = self.decoder2(dec1, enc1, t) # [1, 48, 48, 48, 48]
        dec0 = dec0 + r1
        
        out = self.decoder1(dec0, enc0, t) # [1, 48, 96, 96, 96]
        out = out + r0
        logits = self.out(out) # + comp # add composition to output
        
        # logits = logits + self.vxm(logits, image, noise)
        # logits = self.vxm(logits, image, noise)
        
        return logits
    
    def reverse_attention(self, x: torch.Tensor) -> torch.Tensor:
        crop = -(torch.sigmoid(x)) + 1
        r = torch.mul(x, crop)
        return r

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )
    
    