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

import torch
import torch.nn as nn

from monai.utils import ensure_tuple_rep

from .layers import TwoConv, Down


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
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

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
            
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        return [x0, x1, x2, x3, x4]
        

