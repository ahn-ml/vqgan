import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from einops import rearrange, repeat
import os

num_classes_list = [5, 4, 9, 16]  # Color, pos2x2, pos3x3, pos4x4
use_post_vq = True

DJEPA = True


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """

    def __init__(
        self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1:  # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g["weight_decay"])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (g["trust_coefficient"] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])


class MultiHeadAttentionPool(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        # x: [B, N, D]
        query = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attention(query, x, x)
        return out.squeeze(1)  # [B, D]


class FeatAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # bs, seq_len, dims = x.shape
        x = x.permute((0, 2, 1))
        return self.avg_pool(x).squeeze()


def tokenize_all(tokenizer, dataloader, get_factor=False, n_samples_for_dci=1024):
    global_token_collector = []
    patch_token_collector = []
    global_indices_collector = []

    tokenizer.train(False)
    factors = []
    tokenizer.cuda()
    for batch in tqdm(dataloader):
        if get_factor:
            image, label, mask = batch
        else:
            image = batch
        image = image["image"]
        b = image.size(0)
        # prediction
        if use_post_vq:

            global_tokens, _, res = tokenizer.encode(image.cuda())
            indices = res[-1]
            global_indices_collector.append(
                rearrange(indices, "(b t) h w -> b t (h w)", b=b)
            )
        else:
            if DJEPA:
                global_tokens, patch_tokens = tokenizer.encode(image.cuda())
            else:
                global_tokens = tokenizer.embed(image.cuda(), pool=False)

        if DJEPA:
            global_token_collector.append(
                rearrange(global_tokens, "(b t) d h w -> b t (h w) d", b=b)
            )
        else:
            global_token_collector.append(
                rearrange(global_tokens, "(b t) n d -> b t n d", b=b)
            )
        # patch_token_collector.append(patch_tokens)
        if get_factor:
            factors.append(label)

        torch.cuda.empty_cache()
        if len(global_token_collector) * len(image) > n_samples_for_dci:
            break

    global_token_collector = torch.cat(global_token_collector, dim=0)

    # patch_token_collector = torch.cat(patch_token_collector, dim=0) #.squeeze()
    if use_post_vq:
        global_indices_collector = torch.cat(global_indices_collector, dim=0)
    else:
        global_indices_collector = None
    if get_factor:
        factors = torch.cat(factors, dim=0)
        factors = factors.unsqueeze(1)  # [b t c] -> [b t 1 c]
    else:
        factors = None

    _, _, _, L = global_token_collector.size()  # b t n d
    global_token_concat = global_token_collector.flatten(start_dim=2)

    torch.cuda.empty_cache()

    # print('Global Token:', global_token_collector.size()) # [1152, token_size, num_latent_tokens]
    # print('Global Indices:', global_indices_collector.size())

    return global_token_collector, global_indices_collector, factors


def train_classifier(
    probes,
    aggregator,
    tokenizer,
    val_dataloader,
    lr=0.1,
    num_iterations=20000,
    n_samples=1024,
    logging=False,
    run_name=None,
):
    global_tokens, global_token_indices, factors = tokenize_all(
        tokenizer, val_dataloader, get_factor=True, n_samples_for_dci=n_samples
    )

    train_perc = 0.75
    num_points = global_tokens.shape[0] * global_tokens.shape[1]
    print("num_points=", num_points)

    perm = np.random.permutation(num_points)
    with torch.no_grad():
        x = rearrange(global_tokens, "b t n d -> (b t) n d")[perm].cuda()
        factors = rearrange(factors, "b t c -> (b t) c")[perm]
        x_pooled = aggregator(x)
    x_train = x_pooled[: int(num_points * train_perc)]
    x_test = x_pooled[int(num_points * train_perc) :]
    y_train = factors[: int(num_points * train_perc)].long().cuda()
    y_test = factors[int(num_points * train_perc) :].long().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = LARS(probes.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(probes.parameters(),lr=lr)

    probes.train()
    for iter in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        losses = [
            criterion(probes[i](x_train), y_train[:, i])
            for i in range(len(num_classes_list))
        ]

        loss = sum(losses)
        loss.backward()
        optimizer.step()
        # Log training loss
        if iter % 1000 == 0:

            # Evaluation and logging final results
            with torch.no_grad():
                test_accuracies = []
                for i in range(len(num_classes_list)):
                    output = probes[i](x_test)
                    preds = output.argmax(-1)
                    accuracy = (preds == y_test[:, i]).float().mean()
                    test_accuracies.append(accuracy.item())
                    print(f"Task {i+1} Accuracy: {accuracy.item():.4f}")

            avg_accuracy = sum(test_accuracies) / len(test_accuracies)
            print(f"Average Accuracy: {avg_accuracy:.4f}")


if __name__ == "__main__":
    from main import instantiate_from_config
    import hydra

    from torch.utils.data import DataLoader
    from pytorch_lightning import seed_everything
    from taming.models.vqgan import VQModel

    DATASET_PATH = "/data/hslee/discrete-jepa/runner/datasets/data"

    with hydra.initialize(version_base=None, config_path=f"./configs"):
        cfg = hydra.compose(config_name="custom_vqgan_blink.yaml")
        print(cfg)

    codebook_size = cfg.model.params.n_embed

    tokenizer_ckpt_path = "/data/hslee/taming-transformers/logs/2025-06-19T15-57-53_custom_vqgan_blink/checkpoints/last.ckpt"
    seed_everything(cfg.seed, workers=True)

    def load_tokenizer(cfg, tokenizer_ckpt_path):

        tokenizer = instantiate_from_config(cfg.model)
        tokenizer.load_state_dict(
            torch.load(tokenizer_ckpt_path, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
        )
        tokenizer = tokenizer.eval()
        tokenizer.requires_grad_(False)
        return tokenizer

    tokenizer = load_tokenizer(cfg, tokenizer_ckpt_path)

    # autofinder of default path (only when the path is relative path)

    probe_input_size = 96 * 16
    # probe_input_size = 768
    # Load Validation Dataset
    val_dataloader = DataLoader(
        instantiate_from_config(cfg.data.params.validation),
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    seed_everything(cfg.seed, workers=True)
    # probes = nn.ModuleList(nn.Sequential(nn.Linear(cfg.worldmodel.vq_model.token_size, hidden_dim),
    #                                      nn.ReLU(),
    #                                      nn.Linear(hidden_dim, num_classes)
    #                                      )
    #         for num_classes in num_classes_list).cuda()
    probes = nn.ModuleList(
        nn.Linear(probe_input_size, num_classes) for num_classes in num_classes_list
    ).cuda()
    # aggregator = MultiHeadAttentionPool(dim=cfg.worldmodel.vq_model.token_size, num_heads=1).cuda()
    # aggregator = FeatAvgPool().cuda()
    aggregator = lambda x: torch.flatten(x, start_dim=-2)
    train_classifier(
        probes,
        aggregator,
        tokenizer,
        val_dataloader,
        lr=0.1,
        n_samples=np.inf,
        num_iterations=20000,
        logging=False,
    )
    torch.save(probes.state_dict(), "cls_blink_lr0.1_ep20000_flatten_vqgan_cat.ckpt")
