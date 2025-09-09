import math
import os
from functools import partial

import hydra
import numpy as np
import torch
import torch.distributed as dist
from decoders import CrossAttentionDecoder, CrossAttentionDecoderConfig
from einops import rearrange, repeat
from hydra import compose, initialize
from PIL import Image
from pytorch_lightning import seed_everything
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from vision_transformer import MLP, DropPath, trunc_normal_
from datetime import datetime
import wandb

DATASET_PATH = "/data/hslee/discrete-jepa/runner/datasets/data"
from utils import DistributedEvalSampler
from main import instantiate_from_config


DJEPA = True

invTrans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)
un_normalize = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / torch.tensor(1)),
        transforms.Normalize(mean=-torch.tensor(0), std=[1.0, 1.0, 1.0]),
    ]
)

discrete_colors = [  # K, R, B, C, M, Y, W
    [0, 0, 0],
    [255, 0, 0],
    [0, 0, 255],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
    [255, 255, 255],
]
color_palette = np.array(discrete_colors).astype(np.uint8)


def linear_warmup_with_cos_decay(
    step,
    warmup_start_value,
    warmup_final_value,
    warmup_start_step,
    warmup_final_step,
    total_steps,
    final_value,
):
    assert warmup_start_value <= warmup_final_value
    assert warmup_start_step <= warmup_final_step

    if warmup_start_step <= step < warmup_final_step:
        # interpolate linearly
        a = warmup_final_value - warmup_start_value
        b = warmup_start_value
        progress = (step + 1 - warmup_start_step) / (
            warmup_final_step - warmup_start_step
        )
        value = a * progress + b
        return value
    elif step < warmup_start_step:
        value = warmup_start_value
    elif warmup_final_step <= step:
        value = warmup_final_value
        progress = (step + 1 - warmup_final_step) / (total_steps - warmup_final_step)

        return 0.5 * (
            1.0 + final_value + (1.0 - final_value) * math.cos(math.pi * progress)
        )


def ddp_setup():

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", init_method="env://", rank=rank, world_size=world_size
    )

    device = torch.cuda.current_device()

    if dist.get_rank() == 0:
        print(f"{rank}, {world_size}, {local_rank}")
    return rank, world_size, local_rank, device


def load_tokenizer(cfg, tokenizer_ckpt_path, device):
    tokenizer = instantiate_from_config(cfg.model)
    tokenizer.load_state_dict(
        torch.load(tokenizer_ckpt_path, map_location=device, weights_only=False)[
            "state_dict"
        ]
    )
    tokenizer = tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer


class ReconModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        decoder: CrossAttentionDecoder,
        codebook_size,
        decoder_dim=768,
        decode_indices=False,
        init_mask_type=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.decode_indices = decode_indices
        assert init_mask_type is None or init_mask_type in ["init", "white"]
        self.init_mask_type = init_mask_type
        if self.decode_indices:
            self.emb = nn.Embedding(codebook_size, decoder_dim)

    def forward(self, x):
        """
        x: Images: [B T C H W]

        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        b, t, c, h, w = x.size()
        if DJEPA:
            x_timeflat = rearrange(x, "b t ... -> (b t) ...")
            z_quantized, _, result_dict = self.tokenizer.encode(x_timeflat)
            z_indices = result_dict[-1]
            if self.decode_indices:
                z_enc_out = self.emb(rearrange(z_indices, "(b t) 1 p -> (b t) p", b=b))
            else:
                z_enc_out = rearrange(z_quantized, "(b t) d h w -> (b t) (h w) d", b=b)
            # print(z_enc_out.shape)
        else:
            z_enc_out = self.tokenizer.embed(x, pool=True)
            z_enc_out = rearrange(z_enc_out, "(b t) d -> (b t) 1 d", b=b)

            # consider last output: b p d

        if self.init_mask_type is None:
            recon = rearrange(
                self.decoder(slots=z_enc_out).recon, "(b t) ... -> b t ...", b=b
            )

        elif self.init_mask_type == "init":
            init_mask = rearrange(x, "b t ... -> (b t) ...")
            recon = rearrange(
                self.decoder(slots=z_enc_out, init_mask=init_mask).recon,
                "(b t) ... -> b t ...",
                b=b,
            )
        elif self.init_mask_type == "white":
            init_mask = (
                torch.any(rearrange(x, "b t ... -> (b t) ...") > 0, dim=1) * 1.0
            )[:, None].repeat(1, 3, 1, 1)
            Image.fromarray(
                (init_mask[0].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            ).save("asdfasdfsad.png")
            recon = rearrange(
                self.decoder(slots=z_enc_out, init_mask=init_mask).recon,
                "(b t) ... -> b t ...",
                b=b,
            )

        return recon


def eval_step(model, val_dataloader):
    model.eval()
    y_preds = []
    l = []
    labels = []
    flag = 0
    for batch in tqdm(val_dataloader):
        with torch.no_grad():
            batch, _, _ = batch
            vid = batch["pixel_values"]
            label = batch["pixel_values_discrete"].squeeze()
            y_pred = model(vid.cuda()).squeeze()
            y_preds.append(y_pred)
            labels.append(label.cuda())

    ypreds = torch.cat(y_preds, dim=0).cuda()
    labels = torch.cat(labels, dim=0).cuda()
    print(ypreds.shape)

    return ypreds, labels


def train(
    cfg,
    model,
    criterion,
    train_dataloader,
    val_dataloader,
    postfix="",
    load_ckpt=None,
    logging=True,
):

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{torch.cuda.current_device()}"
    lr = cfg.worldmodel.lr
    if load_ckpt is not None:
        model.load_state_dict(
            torch.load(load_ckpt, map_location="cuda", weights_only=False)
        )
        return model
    if logging and dist.get_rank() == 0:
        wandb.init(
            project="djepa",
            config=dict(cfg),
            name=f"recon_blink_titok_lr{lr}_dev{world_size}_{postfix}",
        )
        wandb.config.update({"lr": lr})

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            partial(
                linear_warmup_with_cos_decay,
                warmup_start_value=0.0,
                warmup_final_value=1.0,
                warmup_start_step=0.0,
                warmup_final_step=cfg.worldmodel.warmup_steps_pct
                * cfg.worldmodel.total_steps,  # will do num_processes times, so need to scale
                total_steps=cfg.worldmodel.total_steps,
                final_value=1e-6 / lr,
            )
        ]
        * len(optimizer.param_groups),
    )
    global_step = 0
    epoch = 0
    while global_step < cfg.worldmodel.total_steps:
        dist.barrier()
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = []
        bar = tqdm(train_dataloader)
        for batch in bar:

            lr = optimizer.param_groups[0]["lr"]
            vid = batch["pixel_values"]
            label = batch["pixel_values_discrete"].squeeze()
            y_pred = model(vid.cuda()).squeeze()
            loss = criterion(y_pred, label.cuda())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss.append(loss.item())
            bar.set_description(f"Train Loss: {loss.item():.5f}")
            if logging and dist.get_rank() == 0:
                wandb.log({"train_loss": loss.item(), "lr": lr}, step=global_step)
            global_step += 1
            del batch, label, vid
            torch.cuda.empty_cache()

        print(f"Epoch {epoch} train_loss={sum(epoch_loss)/len(epoch_loss):.5f}")
        dist.barrier()

        evaluation(
            model,
            val_dataloader,
            criterion,
            logging,
            global_step,
            epoch,
            postfix,
            world_size,
            device,
        )

        dist.barrier()  # ------------------------ barrier ------------------------------
        if (epoch + 1) % 5 == 0:

            torch.save(
                model.state_dict(),
                f"recon_out{postfix}/titok_blink_recon_ep{global_step}{postfix}.ckpt",
            )

        epoch += 1

    if dist.get_rank() == 0 and logging:
        wandb.finish()


def evaluation(
    model,
    val_dataloader,
    criterion,
    logging,
    global_step,
    epoch,
    postfix,
    world_size,
    device,
):
    y_preds, labels = eval_step(model, val_dataloader)
    local_preds_sizes = torch.tensor(y_preds.size(), device=device)
    local_label_sizes = torch.tensor(labels.size(), device=device)
    global_preds_sizes = [
        torch.zeros_like(local_preds_sizes, device=device) for _ in range(world_size)
    ]
    global_label_sizes = [
        torch.zeros_like(local_label_sizes, device=device) for _ in range(world_size)
    ]

    dist.all_gather(global_preds_sizes, local_preds_sizes)
    dist.all_gather(global_label_sizes, local_label_sizes)

    global_ypreds = [
        torch.zeros(tuple(_s), device=device, dtype=y_preds.dtype)
        for _s in global_preds_sizes
    ]
    global_labels = [
        torch.zeros(tuple(_s), device=device, dtype=labels.dtype)
        for _s in global_label_sizes
    ]

    dist.all_gather(global_ypreds, y_preds)
    dist.all_gather(global_labels, labels)

    global_ypreds = torch.cat(global_ypreds).cpu()
    global_labels = torch.cat(global_labels).cpu()

    if dist.get_rank() == 0:

        with torch.no_grad():
            eval_loss = criterion(global_ypreds, global_labels)
        # eval_mAP = 0 #MultilabelAveragePrecision(global_labels.shape[1])(nn.functional.sigmoid(global_ypreds), global_labels)
        # eval_acc_top1 = MulticlassAccuracy(global_ypreds.shape[-1])(global_ypreds,global_labels)
        # eval_acc_top5 = MulticlassAccuracy(global_ypreds.shape[-1],top_k=5)(global_ypreds,global_labels)
        acc = (
            torch.argmax(global_ypreds, dim=1) == global_labels.long()
        ).sum() / global_labels.numel()
        print(f"Epoch {epoch} val_loss={eval_loss:.5f}")
        if logging:
            wandb.log({"pixel_acc": acc}, step=global_step)
            wandb.log({"val_loss": eval_loss}, step=global_step)

        hh = 5
        ww = 8
        _m = 10
        _ms = 5
        imgen = []
        for i, (im, lab) in enumerate(
            zip(global_ypreds[: hh * ww], global_labels[: hh * ww])
        ):
            new_im = color_palette[im.argmax(0).numpy().astype(int)]
            new_lab = color_palette[lab.numpy().astype(int)]
            bar = np.ones((new_im.shape[0], _ms, 3), dtype=np.uint8) * 255
            imgen.append(np.concatenate((new_im, bar, new_lab), axis=1))
            # a = Image.fromarray(np.concatenate((new_im,new_lab),axis=1),mode='RGB')
            # os.makedirs(f'out{postfix}',exist_ok=True)
            # a.save(f'out{postfix}/{epoch+1}_{i}.png')

        allimm = (
            np.ones(
                (
                    (hh + 1) * _m + imgen[0].shape[0] * hh,
                    (ww + 1) * _m + imgen[0].shape[1] * ww,
                    3,
                ),
                dtype=np.uint8,
            )
            * 255
        )
        for i in range(hh):
            for j in range(ww):
                idx = i * ww + j
                if idx >= len(imgen):
                    break
                allimm[
                    _m
                    + i * (_m + imgen[0].shape[0]) : _m
                    + i * (_m + imgen[0].shape[0])
                    + imgen[0].shape[0],
                    _m
                    + j * (_m + imgen[0].shape[1]) : _m
                    + j * (_m + imgen[0].shape[1])
                    + imgen[0].shape[1],
                ] = imgen[idx]
        allimm = Image.fromarray(allimm, mode="RGB")
        os.makedirs(f"recon_out{postfix}", exist_ok=True)
        allimm.save(f"recon_out{postfix}/{epoch+1}_all.png")
        allimm.save(f"recon_out{postfix}/recent_all.png")


def get_postfix(cfg, use_post_vq):

    token_size = str(96)
    num_latent_tokens = str(16)
    vq_mode = f"-V{1024}"
    postfix = f"-T{token_size}-L{num_latent_tokens}{vq_mode}-H1"

    return postfix


def main(cfg, tokenizer, use_post_vq=True):

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    # Load Train Dataset with Labels
    train_dataset = instantiate_from_config(cfg.data.params.train)
    val_dataset = instantiate_from_config(cfg.data.params.validation)
    # Load Validation Dataset

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedEvalSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        sampler=train_sampler,
        num_workers=8,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        sampler=val_sampler,
        num_workers=8,
    )

    # tokenizer.requires_grad_(True)
    # tokenizer.train()
    decode_indices = False  # True if DJEPA else False
    d_model = 96
    d_model_decoder = 64
    init_mask_type = "white"

    if not DJEPA:
        use_post_vq = False

    decoder_cfg = CrossAttentionDecoderConfig(
        input_size=d_model,
        hidden_size=d_model_decoder,
        num_hidden_layers=6,
        num_attention_heads=4,
        hidden_dropout_prob=0.3,
        image_size=64,
        patch_size=16,
        num_channels=7,
        num_cls_tokens=1,
        cls_split_to_slots=False,
        init_mask_type=init_mask_type,
    )
    decoder = CrossAttentionDecoder(decoder_cfg)
    model = ReconModel(
        tokenizer,
        decoder,
        codebook_size=1024,
        decoder_dim=d_model,
        decode_indices=decode_indices,
        init_mask_type=init_mask_type,
    )
    model = DistributedDataParallel(model.cuda(), find_unused_parameters=False)

    # model.load_state_dict(torch.load('/data/hslee/discrete-jepa/out-T96-L32-svq-V1024x1-g2p-postvq-H6_6layer_8head/djepa_blink_recon_ep201000-T96-L32-svq-V1024x1-g2p-postvq-H6.ckpt', map_location='cuda', weights_only=False),strict=True)
    postfix = (
        datetime.now().strftime("%Y%m%d%H%M%S")[2:]
        + get_postfix(cfg, use_post_vq=use_post_vq)
        + f"_{d_model}dim{'_i2im' if decode_indices else '_r2im'}{'_'+init_mask_type if init_mask_type else ''}_vqgan"
    )
    criterion = torch.nn.CrossEntropyLoss()
    train(
        cfg,
        model,
        criterion,
        train_dataloader,
        val_dataloader,
        postfix=postfix,
        logging=True,
    )
    # evaluation(cfg, model, val_dataloader, criterion, logging=False, global_step=0, epoch=0, postfix=postfix, world_size=world_size, device=device)


def config():
    # Load Configuration
    with initialize(version_base=None, config_path=f"./configs"):
        cfg = compose(config_name="custom_vqgan_blink.yaml")

    # cfg.worldmodel.patch_size=8

    cfg.worldmodel.total_steps = 1000 * 50
    cfg.worldmodel.lr = 1e-3

    cfg.data.params.train.params.video_len = 1
    cfg.data.params.validation.params.video_len = 1
    cfg.data.params.train.params.root = "/data/hslee/discrete-jepa/runner/datasets/data/blinking_balls/4Balls_12Frames_Blinking_Pattern"
    cfg.data.params.validation.params.root = "/data/hslee/discrete-jepa/runner/datasets/data/blinking_balls/4Balls_12Frames_Blinking_Pattern"
    cfg.data.params.train.params.stochastic_sample = True
    cfg.data.params.validation.params.stochastic_sample = True

    cfg.data.params.train.params.output_discrete_pixel_value = True
    cfg.data.params.validation.params.output_discrete_pixel_value = True
    if DJEPA:
        tokenizer_ckpt_path = "/data/hslee/taming-transformers/logs/2025-06-19T15-57-53_custom_vqgan_blink/checkpoints/last.ckpt"

    seed_everything(cfg.seed, workers=True)

    tokenizer = load_tokenizer(
        cfg, tokenizer_ckpt_path, f"cuda:{torch.cuda.current_device()}"
    )

    return tokenizer, cfg


if __name__ == "__main__":
    ddp_setup()
    tokenizer, cfg = config()
    main(cfg, tokenizer, use_post_vq=True)
