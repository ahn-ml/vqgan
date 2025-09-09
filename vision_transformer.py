# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from einops import rearrange

from utils import (
    trunc_normal_,
    repeat_interleave_batch,
    apply_masks,
    _expand_token,
)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_dim = ((img_size[0] // patch_size), (img_size[1] // patch_size))
        num_patches = self.patch_dim[0] * self.patch_dim[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class TubletEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, in_time=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_dim = (img_size[0] // patch_size, img_size[1] // patch_size)
        num_patches = self.patch_dim[0] * self.patch_dim[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_time = in_time
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(patch_size, patch_size, in_time), stride=(patch_size, patch_size, in_time))

    def forward(self, x):
        BT, C, H, W = x.shape
        x = rearrange(x, '(b t) c h w -> b c h w t', t=self.in_time)
        x = self.proj(x) # b c 1 h/p w/p
        x = x.flatten(2).transpose(1, 2) # b h/p*w/p c
        return x

class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=(224, 224), in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)*(img_size[1] // stride_prod)
        self.patch_dim = (img_size[0] // stride_prod, img_size[1] // stride_prod)
    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class Conv5x5Embed(nn.Module):
    """
    4x4 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=(224, 224), in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [
                nn.Conv2d(
                    channels[i], channels[i + 1], kernel_size=5, stride=strides[i], padding=2, bias=(not batch_norm)
                )
            ]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i + 1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)*(img_size[1] // stride_prod)
        self.patch_dim = (img_size[0] // stride_prod, img_size[1] // stride_prod)

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)

class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        num_latent_tokens=16,
        token_size=12,
        output_dim=768,
        **kwargs
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        if type(num_patches) is int:
            patch_dim = (int(num_patches**.5), )*2 
            assert patch_dim[0] * patch_dim[1] == num_patches, f"num_patches {num_patches} is not a square number"
        else:
            patch_dim = (num_patches[0], num_patches[1])
            num_patches = num_patches[0] * num_patches[1]
        self.num_patches = num_patches
        self.predictor_global_embed = nn.Linear(token_size, predictor_embed_dim, bias=True)
        self.predictor_patch_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.latent_token_positional_embedding = nn.Parameter(
            (predictor_embed_dim**-0.5) * torch.randn(num_latent_tokens, predictor_embed_dim))
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      patch_dim,
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, output_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z, masks_x, masks, mode='g2p'):
        cond_type = mode[0]
        pred_type = mode[-1]
        
        assert (cond_type in ['g', 'p', 'a']) and (pred_type in ['g', 'p'])
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]
        
        if cond_type == 'g':
            B, C, H, W = z.shape
            assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
            latent_tokens = z.reshape(B, C*H, W).permute(0, 2, 1) # B,L,D
            
            # -- map from encoder-dim to pedictor-dim
            latent_tokens = self.predictor_global_embed(latent_tokens)
            # -- add positional embedding to x tokens
            latent_tokens += self.latent_token_positional_embedding
            
        elif cond_type == 'p':
            latent_tokens = z
            
            # -- map from encoder-dim to pedictor-dim
            latent_tokens = self.predictor_patch_embed(latent_tokens)
            # -- add positional embedding to x tokens
            latent_tokens_pos_embed = self.predictor_pos_embed.repeat(z.size(0), 1, 1)
            latent_tokens += apply_masks(latent_tokens_pos_embed, masks_x)
        elif cond_type == 'a':
            z_g, z_p = z
            B, C, H, W = z_g.shape

            assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
            # assert flattened
            
            global_tokens = z_g.reshape(B, C*H, W).permute(0, 2, 1) # B,L,D
            patch_tokens = z_p

            # -- map from encoder-dim to pedictor-dim
            global_tokens = self.predictor_global_embed(global_tokens)
            patch_tokens = self.predictor_patch_embed(patch_tokens)
            # -- add positional embedding to x tokens
            global_tokens += self.latent_token_positional_embedding
            
            patch_tokens +=  self.predictor_pos_embed.repeat(z_p.size(0), 1, 1)[:, :1, :]
            #patch_tokens_pos_embed = self.predictor_pos_embed.repeat(z_p.size(0), 1, 1)
            #patch_tokens = patch_tokens + apply_masks(patch_tokens_pos_embed, masks_x)

            latent_tokens = torch.cat([global_tokens,patch_tokens],dim=1) # B L D

        B, L, D = latent_tokens.shape
        if pred_type == 'g':
            pos_embs = self.latent_token_positional_embedding.repeat(B*len(masks), 1, 1)
            # --
            pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
            # --
            pred_tokens += pos_embs
        else:
            # -- concat mask tokens to x
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            # --
            pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
            # --
            pred_tokens += pos_embs

        latent_tokens = latent_tokens.repeat(len(masks), 1, 1)
        x = torch.cat([latent_tokens, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, L:]
        x = self.predictor_proj(x)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        num_latent_tokens=16,
        token_size=12,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        if type(img_size) is int:
            img_size = [img_size, img_size]
        # --
        self.latent_tokens = nn.Parameter(torch.zeros(num_latent_tokens, embed_dim))
        
        self.latent_token_positional_embedding = nn.Parameter(
            (embed_dim**-0.5) * torch.randn(num_latent_tokens, embed_dim))
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patch_dim = self.patch_embed.patch_dim
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            patch_dim,
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.conv_out = nn.Conv2d(embed_dim, token_size, kernel_size=1, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.latent_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)
            
        B, M, D = x.shape
        latent_tokens = _expand_token(self.latent_tokens, x.shape[0])
        latent_tokens = latent_tokens + self.latent_token_positional_embedding
        x = torch.cat([x, latent_tokens], dim=1) # B, M+L, D

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        patch_tokens =  x[:, :M] 
        # fake 2D shape
        latent_tokens = x[:, M:]
        latent_tokens = latent_tokens.reshape(B, self.embed_dim, self.num_latent_tokens, 1)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(B, self.token_size, 1, self.num_latent_tokens)

        return patch_tokens, latent_tokens

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}



class TemporalTransformer(nn.Module):
    """Temporal Transformer """
    def __init__(
        self,
        tokenizer,
        num_classes=36,
        dim=384,
        token_size=12,
        num_frames=100,
        num_latent_tokens=16,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        train_tokenizer=False,
        **kwargs
    ):
        super().__init__()
        self.num_frames=num_frames
        #self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.tokenizer = tokenizer
        self.tokenizer.requires_grad_(train_tokenizer)
        self.tokenizer.train(train_tokenizer)
        # --
        self.predictor_global_embed = nn.Linear(token_size, dim, bias=True)

        self.cls_token = nn.Parameter((dim**-0.5) *torch.randn(1, 1, dim))
        self.positional_embedding = nn.Parameter((dim**-0.5) * torch.randn(1, 1, num_latent_tokens, dim//2))

        # --
        self.temporal_embedding = nn.Parameter((dim**-0.5) * torch.randn(1,  num_frames, 1, dim//2))
        #pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1],
        #                                    patch_dim,
        #                                    cls_token=False)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim)
        self.out = nn.Linear(dim, num_classes,bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, use_post_vq=True):
        
        b, t = x.shape[0:2]
        x = rearrange(x, 'b t ... -> (b t) ...')


        if use_post_vq:
            x, _ = self.tokenizer.embed_quant(x)
            x = rearrange(x, '(b t) d 1 p -> b (t p) d', b=b)
        else:
            x, _ = self.tokenizer.embed(x)
            x = rearrange(x, '(b t) p d -> b (t p) d', b=b)
        
        pos = self.positional_embedding.repeat(1,t,1,1)
        tem= self.temporal_embedding.repeat(1,1,self.num_latent_tokens,1)
        enc = rearrange(torch.cat([pos,tem], dim=-1),'1 t p d -> 1 (t p) d')
        x = self.predictor_global_embed(x)
        x = x + enc

        
        x = torch.cat([self.cls_token.repeat(b,1,1),x], dim=1) # B, M+1, D

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)



        cls_out =  x[:, 0] 
        # fake 2D shape
        cls_out = self.out(cls_out)

        return cls_out, x