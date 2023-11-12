import numpy as np
import math

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.layers import trunc_normal_

from .utils import *
from .attention import *
from .embedder import *


class MDTBlock(nn.Module):
    """
    A MDT block with adaptive layer norm zero (adaLN-Zero) conMDTioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, ids_keep=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), ids_keep=ids_keep)
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of MDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
    
class MDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mask_ratio=None,
        decode_layer=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use learnbale sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches) for _ in range(depth)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_patches=num_patches) for _ in range(1)
        ])
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, hidden_size), requires_grad=True)
        if mask_ratio is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.mask_ratio = float(mask_ratio)
            self.decode_layer = int(decode_layer)
        else:
            self.mask_token = nn.Parameter(torch.zeros(
                1, 1, hidden_size), requires_grad=False)
            self.mask_ratio = None
            self.decode_layer = int(decode_layer)
        print("mask ratio:", self.mask_ratio,
              "decode_layer:", self.decode_layer)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in MDT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.sideblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.mask_ratio is not None:
            torch.nn.init.normal_(self.mask_token, std=.02)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_side_interpolater(self, x, c, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, c, ids_keep=None)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x*mask + (1-mask)*x_before

        return x

    def forward(self, x, t, y, enable_mask=False):
        """
        Forward pass of MDT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        enable_mask: Use mask latent modeling
        """
        x = self.x_embedder(
            x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        masked_stage = False

        # masking op for training
        if self.mask_ratio is not None and enable_mask:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, self.mask_ratio)
            masked_stage = True

        for i in range(len(self.blocks)):
            if i == (len(self.blocks) - self.decode_layer):
                if self.mask_ratio is not None and enable_mask:
                    x = self.forward_side_interpolater(x, c, mask, ids_restore)
                    masked_stage = False
                else:
                    # add pos embed
                    x = x + self.decoder_pos_embed

            block = self.blocks[i]
            if masked_stage:
                x = block(x, c, ids_keep=ids_keep)
            else:
                x = block(x, c, ids_keep=None)

        # (N, T, patch_size ** 2 * out_channels)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


    def forward_with_cfg(self, x, t, y, cfg_scale=None, diffusion_steps=1000, scale_pow=4.0):
        """
        Forward pass of MDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        if cfg_scale is not None:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, y)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            scale_step = (
                1-torch.cos(((1-t/diffusion_steps)**scale_pow)*math.pi))*1/2 # power-cos scaling 
            real_cfg_scale = (cfg_scale-1)*scale_step + 1
            real_cfg_scale = real_cfg_scale[: len(x) // 2].view(-1, 1, 1, 1)

            half_eps = uncond_eps + real_cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        else:
            model_out = self.forward(x, t, y)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   MDT Configs                                  #
#################################################################################

def MDT_XL_2(**kwargs):
    return MDT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def MDT_XL_4(**kwargs):
    return MDT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def MDT_XL_8(**kwargs):
    return MDT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def MDT_L_2(**kwargs):
    return MDT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def MDT_L_4(**kwargs):
    return MDT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def MDT_L_8(**kwargs):
    return MDT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def MDT_B_2(**kwargs):
    return MDT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def MDT_B_4(**kwargs):
    return MDT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def MDT_B_8(**kwargs):
    return MDT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def MDT_S_2(**kwargs):
    return MDT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def MDT_S_4(**kwargs):
    return MDT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def MDT_S_8(**kwargs):
    return MDT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)
