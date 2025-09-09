import os
import torch
from torch import nn
from dataclasses import dataclass

from transformers.modeling_utils import PreTrainedModel, ModelOutput
from transformers.configuration_utils import PretrainedConfig
from einops import repeat, rearrange, reduce
import numpy as np
from transformers.models.vit_mae.modeling_vit_mae import get_2d_sincos_pos_embed_from_grid
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import functional as F
from transformers.activations import ACT2FN

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False, num_cls_tokens=1):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate(
            [np.zeros([num_cls_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

class CartesianPositionalEmbedding(nn.Module):
    def __init__(self, channels, image_size):
        super().__init__()
        self.channels = channels
        pos_x = torch.linspace(-1, 1, image_size)
        pos_y = torch.linspace(-1, 1, image_size)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')
        self.register_buffer('grid_x', grid_x.unsqueeze(0))
        self.register_buffer('grid_y', grid_y.unsqueeze(0))
        
    def forward(self, x):
        b, c, h, w = x.shape
        pos_x = repeat(self.grid_x, '1 h w -> b c h w', b=b, c=c//2)
        pos_y = repeat(self.grid_y, '1 h w -> b c h w', b=b, c=c//2)
        return x + torch.cat([pos_x, pos_y], dim=1)



def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    '''
    Use kaiming if next is relu, xavier if next is linear
    '''

    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m

def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):

    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)

    if bias:
        nn.init.zeros_(m.bias)

    return m

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., gain=1.0, inverted=False, norm_over_input=True, epsilon=1e-5, d_model_hidden=None):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.inverted = inverted
        self.norm_over_input = norm_over_input

        self.epsilon = epsilon
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        if d_model_hidden is None:
            d_model_hidden = d_model
        
        self.proj_q = linear(d_model, d_model_hidden, bias=False)
        self.proj_k = linear(d_model, d_model_hidden, bias=False)
        self.proj_v = linear(d_model, d_model_hidden, bias=False)
        self.proj_o = linear(d_model_hidden, d_model, bias=False, gain=gain)
    
    def forward(self, q, k, v, attn_mask=None, attn_bias=None, output_attentions=False):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model·
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_bias is not None:
            attn = attn + attn_bias

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        if self.inverted:
            attn = F.softmax(attn.flatten(start_dim=1, end_dim=2), dim=1).reshape(B, self.num_heads, T, S)
            attn_vis = attn.detach()

            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float('0.'))

            # attn /= attn.sum(dim=-1, keepdim=True) + self.epsilon
            if self.norm_over_input:
                attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
        else:
            attn = F.softmax(attn, dim=-1)
            attn_vis = attn.detach()

        attn = self.attn_dropout(attn)
        
        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)

        outputs = (output, attn_vis) if output_attentions else (output,)
        return outputs

class CrossTransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, max_len, dropout=0., gain=1.,
                intermediate_size=None, intermediate_act='gelu', causal=False,):
        super().__init__()

        intermediate_size = intermediate_size or d_model * 4

        # self.attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, gain=gain)

        if causal:
            mask = torch.triu(torch.ones((max_len, max_len),
                              dtype=torch.bool), diagonal=1)
        else:
            mask = torch.zeros((max_len, max_len), dtype=torch.bool)

        self.self_attn_mask = nn.Parameter(mask, requires_grad=False)

        self.cross_attn_layer_norm = nn.LayerNorm(d_model)
        self.cross_attn_ref_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, gain=gain)

        # self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, intermediate_size, weight_init='kaiming'),
            ACT2FN[intermediate_act],
            linear(intermediate_size, d_model, gain=gain),
            nn.Dropout(dropout))

    def forward(self, input, ref, output_attentions=False):
        """
        input: batch_size x target_len x d_model
        ref: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        T = input.shape[1]
 
        x = self.self_attn_layer_norm(input)
        attn_out = self.self_attn(x, x, x, attn_mask=self.self_attn_mask[:T, :T], output_attentions=output_attentions)
        input = input + attn_out[0]

        x = self.cross_attn_layer_norm(input)
        ref = self.cross_attn_ref_norm(ref)
        attn_out = self.cross_attn(x, ref, ref, output_attentions=output_attentions)
        input = input + attn_out[0]

        x = self.ffn_layer_norm(input)
        x = self.ffn(x)

        output = input + x

        outputs = (output,) + attn_out[1:]

        return outputs

@dataclass
class CrossTransformerOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class CrossTransformer(nn.Module):

    def __init__(self, num_blocks, d_model, num_heads, intermediate_size=None,
                 intermediate_act='gelu', dropout=0., max_len=None, causal=False):
        super().__init__()

        if num_blocks > 0:
            gain = (2 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [
                    CrossTransformerBlock(d_model=d_model, num_heads=num_heads, intermediate_size=intermediate_size,
                                          intermediate_act=intermediate_act, dropout=dropout, gain=gain, max_len=max_len,
                                          causal=causal)
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.blocks = nn.ModuleList()

        # self.layer_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, ref, output_attentions=False):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        all_self_attentions = () if output_attentions else None
        input = self.norm(input)
        for block in self.blocks:
            block_out = block(input, ref, output_attentions=output_attentions)
            input = block_out[0]
            if output_attentions:
                all_self_attentions += (block_out[1],)

        return CrossTransformerOutput(
            last_hidden_state=input,
            attentions=all_self_attentions
        )


class CrossAttentionDecoderConfig(PretrainedConfig):
    model_type = "mult_cls_vit"

    def __init__(
        self,
        input_size=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=None,
        hidden_act="relu",  # should try gelu
        hidden_dropout_prob=0.0,
        image_size=224,
        patch_size=16,
        num_channels=3,
        initializer_range=0.02,
        image_norm_mode="zero_one",
        num_cls_tokens=1,
        cls_split_to_slots=False,
        init_mask_type=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        if input_size is None:
            self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        if intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.image_norm_mode = image_norm_mode
        self.num_cls_tokens = num_cls_tokens
        self.num_patches = (image_size // patch_size) ** 2
        self.cls_split_to_slots = cls_split_to_slots
        self.init_mask_type=init_mask_type


@dataclass
class CrossAttentionDecoderOutput(ModelOutput):
    recon: Optional[Any] = None
    attentions: Optional[Any] = None


class CrossAttentionDecoder(PreTrainedModel):

    config_class = CrossAttentionDecoderConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)


        self.decoder = CrossTransformer(num_blocks=config.num_hidden_layers,
                                        d_model=config.hidden_size,
                                        num_heads=config.num_attention_heads,
                                        dropout=config.hidden_dropout_prob,
                                        intermediate_act=config.hidden_act,
                                        intermediate_size=config.intermediate_size,
                                        causal=False,
                                        max_len=config.num_patches)
        
        self.init_mask_type = config.init_mask_type
        if self.init_mask_type:
            self.patchify = nn.Conv2d(3, config.hidden_size,kernel_size=config.patch_size, stride=config.patch_size)
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # self.position_embeddings = nn.Parameter(
        #     torch.zeros(1, config.num_cls_tokens + config.num_patches, config.hidden_size), requires_grad=False
        # )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.num_patches, config.hidden_size), requires_grad=False
        )
        self.pred = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            linear(config.hidden_size, (config.patch_size ** 2) * config.num_channels, bias=True),
        )
        # if config.input_size != config.hidden_size:
        #     self.input_proj = nn.Linear(config.input_size, config.hidden_size)
        # else:
        #     self.input_proj = None

        self.input_proj = linear(config.input_size, config.hidden_size)

        if config.cls_split_to_slots:
            self.input_decomposer = linear(config.input_size, config.input_size * config.num_cls_tokens)
        
        self.initialize_weights(num_cls_tokens=config.num_cls_tokens)
        
    def initialize_weights(self, num_cls_tokens=1):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(
                self.config.num_patches**0.5),
            add_cls_token=False,
            num_cls_tokens=num_cls_tokens
        )
        self.position_embeddings.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mask_token'}

    def forward(self, slots, output_attentions=False,init_mask=None):
        B, N, D = slots.shape

        if self.init_mask_type:
            B, C, H, W = init_mask.size()
            mask_tokens = self.patchify(init_mask).flatten(2).transpose(1,2)
        else:  
            mask_tokens = self.mask_token.repeat(
                B, self.config.num_patches, 1
            )

        if self.config.cls_split_to_slots:
            slots = self.input_decomposer(slots)
            slots = rearrange(slots, "b 1 (n c) -> b n c", n=self.config.num_cls_tokens)

        # if self.config.input_size != self.config.hidden_size:
        #     slots = self.input_proj(slots)

        slots = self.input_proj(slots)
        hidden_states = mask_tokens + self.position_embeddings

        decoder_outputs = self.decoder(input=hidden_states, ref=slots, output_attentions=output_attentions)
        last_hidden_state = decoder_outputs.last_hidden_state

        # hidden_states = torch.cat([slots, mask_tokens], dim=1)
        # hidden_states = hidden_states + self.position_embeddings

        # decoder_outputs = self.encoder(
        #     hidden_states,
        #     output_attentions=output_attentions,
        # )
        # hidden_states = decoder_outputs.last_hidden_state
        # slots = hidden_states[:, :self.config.num_cls_tokens, :]
        # last_hidden_state = hidden_states[:, self.config.num_cls_tokens:, :]

        logits = rearrange(
            self.pred(last_hidden_state), "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=int(self.config.num_patches**0.5), w=int(self.config.num_patches**0.5), 
            p1=self.config.patch_size, p2=self.config.patch_size, 
            c=self.config.num_channels
        )

        return CrossAttentionDecoderOutput(
            recon=logits,
            attentions=decoder_outputs.attentions,
        )
