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

from typing import Optional, Sequence, Union

import copy
from einops import rearrange

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

from .utils import mask_func
from .utils import get_mask_labels, get_mask_labelsv2


class TwoConv(nn.Sequential):
    """two convolutions."""

    # @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    # @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    # @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x


class BasicUNet(nn.Module):
    # @deprecated_arg(
    #     name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    # )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.1,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
        pool_size = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        select_reconstruct_region=[[4, 4, 4], [12, 12, 12]],
        first_level_region = (32, 32, 32),
        pretrained=True,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions
            
        # additional settings for HybridMIM
        deepth = len(pool_size)
        self.deepth = deepth
        self.in_channels = in_channels
        self.select_reconstruct_region = select_reconstruct_region
        self.pretrained = pretrained
        
        self.stages = self.cons_stages(pool_size, select_reconstruct_region)
        self.pool_size_all = self.get_pool_size_all(pool_size)
        self.window_size = torch.tensor(first_level_region) // torch.tensor(self.pool_size_all)

        # BasicUNet settings
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        
        # downsample layers(encoder)
        self.down = nn.ModuleList([])
        for d in range(deepth):
            self.down.append(Down(3, fea[d], fea[d+1], act=act, norm=norm, bias=bias, dropout=dropout))
        
        # upsample layers(decoder)
        self.up = nn.ModuleList([])
        for d in range(deepth):
            self.up.append(UpCat(3, fea[deepth-d], fea[deepth-d-1], fea[deepth-d-1], 
                                  act, norm, bias, dropout, upsample=upsample))

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        self.decoder_pred = nn.Conv3d(fea[0], out_channels, 1, 1)
        
        # additional settings for HybridMIM
        if pretrained:
            bottom_feature = features[-1]
            self.pred_mask_region = nn.Linear(bottom_feature, 9)# 一个region 4个 patch
            self.contrast_learning_head = nn.Linear(bottom_feature, 384)
            self.pred_mask_region_position = nn.Linear(bottom_feature, 8)
        
    def cons_stages(self, pools, region):
        stage = [(copy.deepcopy(region[0]), copy.deepcopy(region[1]))]
        for pool in reversed(pools):
            for i, r in enumerate(region):
                region[i][0] = region[i][0] * pool[0]
                region[i][1] = region[i][1] * pool[1]
                region[i][2] = region[i][2] * pool[2]
            stage.append((copy.deepcopy(region[0]), copy.deepcopy(region[1])))

        return stage
    
    def get_pool_size_all(self, pool_size):
        p_all = [1, 1, 1]
        for p in pool_size:
            p_all[0] = p_all[0] * p[0]
            p_all[1] = p_all[1] * p[1]
            p_all[2] = p_all[2] * p[2]
        return p_all 

    def wrap_feature_selection(self, feature, region_box):
        # feature: b, c, d, w, h
        return feature[..., region_box[0][0]:region_box[1][0], region_box[0][1]:region_box[1][1], region_box[0][2]:region_box[1][2]]

    def get_local_images(self, images):
        images = self.wrap_feature_selection(images, region_box=self.stages[-1])
        return images
    
    def forward_encoder(self, x):
        x = self.conv_0(x)
        x_down = [x]
        for d in range(self.deepth):
            x = self.down[d](x)
            x_down.append(x)
        return x_down

    def forward_decoder(self, x_down):
        x = self.wrap_feature_selection(x_down[-1], self.stages[0])

        for d in range(self.deepth):
            x = self.up[d](x, self.wrap_feature_selection(x_down[self.deepth-d-1], self.stages[d+1]))
        logits = self.decoder_pred(x)
        return logits

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        device = x.device
        images = x.detach()
        local_images = self.get_local_images(images)
        if self.pretrained:
            # mask_ratio = torch.clamp(torch.rand(1), 0.4, 0.75)
            mask_ratio = 0.4
            x, mask = mask_func(x, self.in_channels, mask_ratio, (16, 16, 16), (6, 6, 6), mask_value=0.0)
            region_mask_labels = get_mask_labels(x.shape[0], 3*3*3, mask, 2*2*2, device)
            region_mask_position = get_mask_labelsv2(x.shape[0], 3*3*3, mask, 2*2*2, device=device)

            x_mask = self.wrap_feature_selection(x, region_box=self.stages[-1])

        hidden_states_out = self.forward_encoder(x)
        logits = self.forward_decoder(hidden_states_out)  

        if self.pretrained:
            # print(hidden_states_out.shape)
            classifier_hidden_states = rearrange(hidden_states_out[-1], "b c (d m) (w n) (h l) -> b c d w h (m n l)", m=self.window_size[0], n=self.window_size[1], l=self.window_size[2])
            classifier_hidden_states = classifier_hidden_states.mean(dim=-1)
            with torch.no_grad():
                hidden_states_out_2 = self.forward_encoder(x)
            encode_feature = hidden_states_out[-1]
            encode_feature_2 = hidden_states_out_2[-1]

            x4_reshape = encode_feature.flatten(start_dim=2, end_dim=4)
            x4_reshape = x4_reshape.transpose(1, 2)

            x4_reshape_2 = encode_feature_2.flatten(start_dim=2, end_dim=4)
            x4_reshape_2 = x4_reshape_2.transpose(1, 2)

            contrast_pred = self.contrast_learning_head(x4_reshape.mean(dim=1))
            contrast_pred_2 = self.contrast_learning_head(x4_reshape_2.mean(dim=1))

            pred_mask_feature = classifier_hidden_states.flatten(start_dim=2, end_dim=4)
            pred_mask_feature = pred_mask_feature.transpose(1, 2)
            mask_region_pred = self.pred_mask_region(pred_mask_feature)

            pred_mask_feature_position = classifier_hidden_states.flatten(start_dim=2, end_dim=4)
            pred_mask_feature_position = pred_mask_feature_position.transpose(1, 2)
            mask_region_position_pred = self.pred_mask_region_position(pred_mask_feature_position)

            return {
                "logits": logits,
                'images': local_images,
                "pred_mask_region": mask_region_pred,
                "pred_mask_region_position": mask_region_position_pred,
                "mask_position_lables": region_mask_position,
                "mask": mask,
                "x_mask": x_mask,
                "mask_labels": region_mask_labels,
                "contrast_pred_1": contrast_pred,
                "contrast_pred_2": contrast_pred_2,
            }
        else :
            return logits


BasicUnet = Basicunet = basicunet = BasicUNet


class BasicUNetEncoder(nn.Module):
    # @deprecated_arg(
    #     name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    # )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (64, 64, 128, 256, 512, 64),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down = nn.ModuleList()
        for d in range(len(fea[:4])):
            self.down.append(Down(spatial_dims, fea[d], fea[d+1], act, norm, bias, dropout))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        _x = [self.conv_0(x)]
        for i in range(len(self.down)):
            _x.append(self.down[i](_x[i]))

        return _x
        

