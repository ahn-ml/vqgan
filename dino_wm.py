# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange, repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


def generate_mask_matrix_min_window(npatch, nwindow, min_context=4):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        if i + 1 < min_context:
            row = torch.cat([zeros] * (nwindow), dim=1)
        else:
            row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_patches,
        num_frames,
        heads=8,
        dim_head=64,
        dropout=0.0,
        causal=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.bias = generate_mask_matrix(num_patches, num_frames).to("cuda")
        self.causal = causal

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        if self.causal:
            dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self, dim, num_patches, num_frames, depth, heads, dim_head, mlp_dim, dropout=0.0
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            num_patches,
                            num_frames,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTPredictor(nn.Module):
    """
    Transition prediction, predicts next latent vector
    """

    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        proj_in=None,
    ):
        super().__init__()
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # update params for adding causal attention masks

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * (num_patches), dim)
        )  # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, num_patches, num_frames, depth, heads, dim_head, mlp_dim, dropout
        )
        self.pool = pool

    def forward(self, x):  # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class ViTCategoricalPredictor(nn.Module):
    """
    Transition prediction, predicts next index of codebook.
    """

    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        codebook_size,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        proj_in=None,
    ):
        super().__init__()
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # update params for adding causal attention masks

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * (num_patches), dim)
        )  # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, num_patches, num_frames, depth, heads, dim_head, mlp_dim, dropout
        )
        self.pool = pool
        self.category = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, codebook_size),
        )
        self.proj_in = proj_in
        if proj_in:
            self.proj = nn.Linear(proj_in, dim)

    def forward(self, x):
        b, n, _ = x.shape
        if self.proj_in:
            x = self.proj(x)
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.category(x)
        return x


class ViTIndex2IndexPredictor(nn.Module):
    """
    Transition prediction, predicts next index of codebook.
    """

    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        codebook_size,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # update params for adding causal attention masks

        self.rulebook = nn.Embedding(codebook_size, dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * (num_patches), dim)
        )  # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, num_patches, num_frames, depth, heads, dim_head, mlp_dim, dropout
        )
        self.pool = pool
        self.category = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, codebook_size),
        )

    def forward(self, x):
        b, n, _ = x.shape
        x = self.rulebook(x.squeeze(-1).long())
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.category(x)
        return x


# adapted from https://github.com/gaoyuezhou/dino_wm/blob/main/models/visual_world_model.py
from torchvision import transforms
import torch.nn.functional as F


class VWorldModel(nn.Module):
    def __init__(
        self,
        cfg,
        use_post_vq,
        encoder,
        predictor,
        num_hist,
        num_pred,
        train_encoder=False,
        train_predictor=True,
        input_indices=False,
        djepa=True,
        djepa_noquant=False,
    ):
        super().__init__()
        # wm_cfg = cfg.worldmodel
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.use_post_vq = use_post_vq
        self.use_ce_loss = use_post_vq
        self.djepa = djepa
        self.djepa_noquant = djepa_noquant
        self.input_indices = input_indices

        # image_size = cfg.dataset.image_size

        self.encoder_transform = lambda x: x
        self.emb_criterion = nn.MSELoss()
        self.emb_criterion_Q = nn.CrossEntropyLoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()

    def encode(self, obs):
        """
        input :  obs (dict): "visual", (b, num_frames, 3, img_size, img_size)
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        z = z_dct["visual"]  # (b, num_frames, num_patches, dim)
        return z

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual",  (b, t, 3, img_size, img_size)
        output:   z (dict): "visual",  (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs["visual"]
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)

        if self.use_post_vq:
            # visual_embs, res = self.encoder.embed(visual)
            visual_embs, _, res = self.encoder.encode(visual)
            indices = res[-1]  # b t h w
            visual_embs = rearrange(
                visual_embs, "(b t) d h w -> b t (h w) d", b=b
            )  # b t n d

            indices = rearrange(indices, "(b t) h w -> b t (h w)", b=b)
            return {"visual": visual_embs, "visual_indices": indices}
        # else:
        #     if self.djepa:
        #         visual_embs, patch_tokens = self.encoder.embed(visual)
        #         visual_embs = rearrange(visual_embs, "(b t) d 1 p -> b t p d", b=b)
        #     else:
        #         visual_embs = self.encoder.embed(visual, pool=False)
        #         visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        #     return {"visual": visual_embs}

    def predict(self, z):  # embedding -> embedding OR embedding -> codebook index
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def forward(self, obs, z_dct=None):
        """
        input:  obs (dict):  "visual" (b, num_frames, 3, img_size, img_size)
                z_dct (dict): "visual"
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_pred, 3, img_size, img_size)
                (unused)visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        # z = self.encode(obs)
        if z_dct is None:
            z_dct = self.encode_obs(obs)
        z = z_dct["visual"]

        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[
            :, self.num_hist : self.num_hist + self.num_pred, :, :
        ]  # (b, num_pred, num_patches, dim)

        if self.use_post_vq:  # this is used for cross entropy loss
            z_idx = z_dct["visual_indices"]  # (b, video_len, num_patches)
            z_idx_src = z_idx[:, : self.num_hist, :]  # (b, num_hist, num_patches)
            z_idx_tgt = z_idx[
                :, self.num_hist : self.num_hist + self.num_pred, :
            ]  # (b, num_pred, num_patches)
            if self.input_indices:
                z_src = z_idx_src.unsqueeze(-1)
        # for decoder target, not used for now
        # visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        # visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:

            visual_pred = None
            z_pred = self.predict(z_src)
            z_pred = z_pred[:, : self.num_pred]
            if (
                self.use_post_vq and self.use_ce_loss
            ):  # if worldmodel predict next codebook index to

                z_idx_loss = self.emb_criterion_Q(
                    z_pred.reshape(-1, z_pred.size(-1)), z_idx_tgt.reshape(-1).long()
                )
                loss_components["z_loss_ce"] = z_idx_loss
                loss_components["z_tgt"] = z_idx_tgt.long()
                loss += z_idx_loss
            else:
                z_loss = self.emb_criterion(z_pred, z_tgt.detach())
                loss_components["z_loss_mse"] = z_loss
                loss += z_loss
            # loss_components["z_visual_loss"] = z_visual_loss
            # loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        visual_reconstructed = None
        loss_components["loss"] = loss
        return (
            z_pred,
            loss,
            loss_components,
        )  # visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):  # unused
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim :] = act_repeated
        return z

    def rollout(self, obs_0, predict_t=3, inc=1, output_indices=False, z_dct=None):
        """
        input:  obs_0 (dict): visual: (b, t, 3, img_size, img_size)

        output: embeddings of rollout obs

                z: (b, t+n+1, num_patches, emb_dim)
        """
        if self.djepa and (not self.input_indices) and self.use_post_vq:
            codebook = self.encoder.quantize.get_codebook_entry
        num_obs_init = obs_0["visual"].shape[1]
        # act_0 = act[:, :num_obs_init]
        # action = act[:, num_obs_init:]
        if z_dct is None:
            z_dct = self.encode_obs(obs_0)  # , act_0)

        # if output_indices and not self.use_post_vq:
        #    raise ValueError()

        indices = None
        if self.use_post_vq:
            if self.input_indices:
                z = z_dct["visual_indices"].unsqueeze(-1)
            else:
                z = z_dct["visual"]
                if self.djepa:
                    indices = z_dct["visual_indices"].unsqueeze(-1)
        else:
            z = z_dct["visual"]

        t = 0
        while t < predict_t:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, :inc, ...]  # t+1, (b inc p d)

            if self.use_post_vq:
                z_new = z_new.argmax(-1)
                if self.input_indices:  # input & output are indices => keep.
                    z_new = z_new.unsqueeze(-1)
                else:  # input is embedding, output is indices => quantize.
                    indices = torch.cat([indices, z_new.unsqueeze(-1)], dim=1)
                    z_new = rearrange(
                        codebook(z_new.flatten()),
                        "(b t p) d -> b t p d",
                        b=len(z_new),
                        t=inc,
                    )
            else:  # input & output are embeddings => keep.
                pass
            z = torch.cat([z, z_new], dim=1)
            t += inc

        # z_pred = self.predict(z[:, -self.num_hist :])
        # z_new = z_pred[:, -1 :, ...] # take only the next pred
        # z = torch.cat([z, z_new], dim=1)
        if output_indices:
            return indices

        return z
