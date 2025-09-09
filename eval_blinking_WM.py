import math
import os
from functools import partial
from datetime import datetime
import hydra
import numpy as np
import torch
import torch.distributed as dist

from einops import rearrange, repeat
from hydra import compose, initialize
from PIL import Image
from pytorch_lightning import seed_everything
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from utils import DistributedEvalSampler
import wandb

DATASET_PATH = "/data/hslee/discrete-jepa/runner/datasets/data"
from taming.models.vqgan import VQModel
import pickle
from lpips import LPIPS

from main import instantiate_from_config

DJEPA = True
AUTOREGRESSIVE = False
AR_PRED = 1000
# utils
invTrans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
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
torch_palette = rearrange(
    torch.tensor(discrete_colors).float() / 255.0, "col c -> 1 col c 1 1"
)


def colorquantizer(tensor):
    # tensor: b c h w [0, 1]
    return (torch_palette.cuda() - tensor.unsqueeze(1)).abs().sum(2).argmin(1)


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


def load_tokenizer(cfg, tokenizer_ckpt_path, map_location):
    tokenizer = instantiate_from_config(cfg.model)
    tokenizer.load_state_dict(
        torch.load(tokenizer_ckpt_path, weights_only=False, map_location=map_location)[
            "state_dict"
        ]
    )
    tokenizer = tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer


def eval_step_ar(
    model,
    probes,
    decoder,
    val_dataloader,
    num_ar_pred,
    emb=None,
    lp_func: LPIPS = None,
    rank=None,
):

    color_palette_torch = torch.tensor(discrete_colors) / 255.0 * 2.0 - 1.0
    color_palette_torch = color_palette_torch.cuda()
    model.eval()
    probes.eval()
    decoder.eval()
    losses = []
    eval_index_acc = []
    if DJEPA:
        codebook = model.module.encoder.quantize.get_codebook_entry
    num_pred = model.module.num_pred
    num_hist = model.module.num_hist
    test_accuracies = []

    lcm = 60
    rep = (num_hist + num_ar_pred) // lcm + 1
    print("rep:", rep)
    print("hist:", num_hist)
    print("pred:", num_ar_pred)
    assert lcm % num_hist == 0
    pick = [0, 10, 100, 999]

    hbar = np.ones((6, (64 + 3) * num_ar_pred, 3), dtype=np.uint8) * 255
    hbar_short = np.ones((6, (64 + 3) * len(pick), 3), dtype=np.uint8) * 255
    vbar_int = torch.ones((num_ar_pred, 64, 3)).long().cuda() * 6
    vbar_short = torch.ones((len(pick), 64, 3)).long().cuda() * 6
    vbar_int_cond = torch.ones((num_hist, 64, 3)).long().cuda() * 6
    skip = 0
    for bid, batch in enumerate(tqdm(val_dataloader)):
        if skip:
            skip -= 1
            continue
        with torch.no_grad():
            batch, _, _ = batch

            B, T, C, H, W = batch["pixel_values"].size()

            # extend batch to infinity -------------------------------------

            # answers (60 period)
            rep_pix = batch["pixel_values_discrete"].cuda().long()  # b t h w
            pixel_values_discrete_extended = rearrange(
                torch.cat((rep_pix[:, num_hist:], rep_pix[:, :num_hist]), dim=1),
                "b t 1 ... -> (b t) ...",
            )
            bg_mask = rearrange(
                (pixel_values_discrete_extended > 0), "(b t) h w -> b t (h w)", b=B
            )
            im1 = color_palette_torch[pixel_values_discrete_extended].permute(
                0, 3, 1, 2
            )

            acc_cont = []
            acc_nobg_cont = []
            lpips_cont = []
            mse_cont = []
            conditioning = batch["pixel_values"][:, :num_hist].cuda()

            for i_ in range(rep):
                z_pred = model.module.rollout(
                    {"visual": conditioning},
                    predict_t=lcm,
                    inc=num_hist,
                    output_indices=False,
                    z_dct=None if i_ == 0 else {"visual_indices": conditioning},
                )
                z_pred = z_pred[:, num_hist : num_hist + lcm]  # b t n #v

                if DJEPA:  # output is index
                    # worldmodel was (repr|idx) -> idx
                    z_pred_idx = z_pred[..., 0]
                    conditioning = z_pred[:, -num_hist:, :, 0]  # last hist (index)
                recon = invTrans(
                    model.module.encoder.decode_tokens(
                        rearrange(z_pred_idx, "b t ... -> (b t) ...")
                    )
                ).clamp(0, 1)
                recon = colorquantizer(recon)  # b t h w
                res = rearrange(
                    recon == pixel_values_discrete_extended,
                    "(b t) h w -> b t (h w)",
                    b=B,
                )

                acc_t = res.sum(-1) / res.size(-1)  # b t
                acc_t_nobg = (res * bg_mask).sum(-1) / bg_mask.sum(-1)  # b t

                acc_cont.append(acc_t)
                acc_nobg_cont.append(acc_t_nobg)

                im0 = color_palette_torch[recon].permute(0, 3, 1, 2)
                l = rearrange(lp_func(im0, im1).squeeze(), "(b t) -> b t", b=B)
                m = rearrange(
                    (((im0 - im1) * 0.5) ** 2).flatten(start_dim=1).mean(-1),
                    "(b t) -> b t",
                    b=B,
                )
                lpips_cont.append(l)
                mse_cont.append(m)

            acc_cont = torch.cat(acc_cont, dim=-1)
            acc_nobg_cont = torch.cat(acc_nobg_cont, dim=-1)
            lpips_cont = torch.cat(lpips_cont, dim=-1)  # b t  concat at t
            mse_cont = torch.cat(mse_cont, dim=-1)
            # all  shapes are : batch ar_pred
            # print(acc_cont.shape, acc_nobg_cont.shape, lpips_cont.shape, mse_cont.shape)
            test_accuracies.append(
                torch.stack((acc_cont, acc_nobg_cont, lpips_cont, mse_cont), dim=-1)
            )  # b t 4

            losses.append(0)
            eval_index_acc.append(torch.zeros((B, num_ar_pred)))

            # r = (loss_components['z_tgt'].detach()==z_pred_idx)
            # eval_index_acc.append(r.sum(-1)/r.size(-1))  # b t

            # quantize the predicted indices  b t n -> (b t n) -> (b t n) d -> (b t) n d
            # z_pred_pooled = torch.flatten(z_pred_quantized, start_dim=1) # (b t) (n d)

            # label_last = label[:, num_hist:num_hist+num_pred].cuda()
            # ta = []
            # for i, probe in enumerate(probes):
            #     output = probe(z_pred_pooled)
            #     preds = rearrange(output.argmax(-1),'(b t) -> b t', b=b)
            #     accuracies = (preds == label_last[:, :, i]).float()
            #     t_ta = []
            #     for ts in range(num_pred):
            #         t_ta.append(accuracies[:, ts].mean().item())
            #     ta.append(t_ta)
            # test_accuracies.append(ta)

            # test_accuracies.append(
            #     torch.stack(
            #         (
            #             acc_t,
            #             acc_t_nobg,
            #         ),
            #         dim=-1,
            #     )
            # )  # b t 2

            # recon = recon.view(B, num_ar_pred,3, H, W)
            # gt = pixel_values_discrete_extended.view(B, num_ar_pred,3, H, W)

            # md = "djepa" if DJEPA else "ijepa"
            # with open(f"figs/{md}/rollout_{md}_recon_{bid}.pkl", "wb") as d:
            #     pickle.dump(color_palette[recon.cpu().numpy()].astype(np.uint8), d)
            # with open(f"figs/{md}/rollout_{md}_target_{bid}.pkl", "wb") as d:
            #     pickle.dump(color_palette[gt.cpu().numpy()].astype(np.uint8), d)
            # with open(f"figs/{md}/rollout_{md}_condition_{bid}.pkl", "wb") as d:
            #     pickle.dump(color_palette[cond_discrete.cpu().numpy()].astype(np.uint8), d)

            # for i in range(B):
            #     ex_img = np.concatenate(
            #             (
            #                 color_palette[rearrange(torch.cat((recon[i],vbar_int),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8),
            #                 hbar,
            #                 color_palette[rearrange(torch.cat((gt[i],vbar_int),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8),

            #             ),
            #                 axis=0
            #             )
            #     ex_short_img = np.concatenate(
            #             (
            #                 color_palette[rearrange(torch.cat((recon[i],vbar_short),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8),
            #                 hbar_short,
            #                 color_palette[rearrange(torch.cat((gt[i],vbar_short),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8),

            #             ),
            #                 axis=0
            #             )
            #     cond_img = color_palette[rearrange(torch.cat((cond_discrete[i],vbar_int_cond),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8)
            #     Image.fromarray(
            #         ex_img
            #     ).save(f"figs/{'djepa' if DJEPA else 'ijepa'}/a_wm_{bid:04d}_{i:02d}_rank{rank}.png")
            #     Image.fromarray(
            #         ex_short_img
            #     ).save(f"figs/{'djepa' if DJEPA else 'ijepa'}/a_wm_{bid:04d}_{i:02d}_rank{rank}_{pick}.png")
            #     Image.fromarray(
            #         cond_img
            #     ).save(f"figs/{'djepa' if DJEPA else 'ijepa'}/a_wm_{bid:04d}_{i:02d}_rank{rank}_cond.png")

            del (
                batch,
                pixel_values_discrete_extended,
                recon,
                z_pred,
                rep_pix,
                z_pred_idx,
                bg_mask,
                res,
                # z_decoder_in,
            )  # ,image_batch, target_batch,ex_img,ex_short_img,cond_img,
            # if decoder.init_mask_type == "white":
            #    del _valid, init_mask_
            torch.cuda.empty_cache()

    losses = torch.tensor(losses).cuda()
    eval_index_acc = torch.cat(eval_index_acc, dim=0).cuda()
    test_accuracies = torch.cat(test_accuracies, dim=0).cuda()  # [b...b] t 4
    print(test_accuracies.shape)
    return losses, eval_index_acc, test_accuracies


def eval_step(model, probes, val_dataloader):
    model.eval()
    losses = []
    eval_index_acc = []
    if DJEPA:
        codebook = model.module.encoder.quantize.embedding
    num_pred = model.module.num_pred
    num_hist = model.module.num_hist
    test_accuracies = []
    for batch in tqdm(val_dataloader):
        with torch.no_grad():
            batch, label, _ = batch
            z_pred, loss, loss_components = model(
                {"visual": batch["pixel_values"].cuda()}
            )
            losses.append(loss)
            # z_pred: DJEPA b t n #v
            # z_pred: IJEPA b t n d
            b = len(z_pred)
            if DJEPA:
                z_pred_idx = torch.argmax(z_pred, dim=-1)  # b t n
                r = loss_components["z_tgt"].detach() == z_pred_idx
                eval_index_acc.append(r.sum(-1) / r.size(-1))  # b t

                # quantize the predicted indices  b t n -> (b t n) -> (b t n) d -> (b t) n d
                z_pred_quantized = rearrange(
                    codebook(z_pred_idx.flatten()),
                    "(b t n) d -> (b t) n d",
                    b=b,
                    t=num_pred,
                )
                z_pred_pooled = torch.flatten(
                    z_pred_quantized, start_dim=1
                )  # (b t) (n d)

            else:
                eval_index_acc.append(torch.zeros(z_pred.shape[0:2]))  # b t
                z_pred = rearrange(z_pred, "b t n d -> (b t) n d")
                z_pred_pooled = torch.flatten(z_pred, start_dim=1)  # (b t) (n d)

            label_last = label[:, num_hist : num_hist + num_pred].cuda()
            ta = []
            for i, probe in enumerate(probes):
                output = probe(z_pred_pooled)
                preds = rearrange(output.argmax(-1), "(b t) -> b t", b=b)
                accuracies = (preds == label_last[:, :, i]).float()
                t_ta = []
                for ts in range(num_pred):
                    t_ta.append(accuracies[:, ts].mean().item())
                ta.append(t_ta)
            test_accuracies.append(ta)
    losses = torch.tensor(losses).cuda()
    eval_index_acc = torch.cat(eval_index_acc, dim=0).cuda()
    test_accuracies = (
        torch.tensor(test_accuracies).cuda().transpose(-1, -2).contiguous()
    )
    return losses, eval_index_acc, test_accuracies


def train(
    cfg,
    model,
    probes,
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
            project="lightning_logs",
            config=dict(cfg),
            name=f"wm_vqgan_blink_lr{lr}_dev{world_size}_{postfix}",
        )
        wandb.config.update({"lr": lr, "gpus": world_size})

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
                final_value=cfg.worldmodel.final_lr / lr,
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

            z_pred, loss, loss_components = model(
                {"visual": batch["pixel_values"].cuda()}
            )

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
            del batch
            torch.cuda.empty_cache()

        print(f"Epoch {epoch} train_loss={sum(epoch_loss)/len(epoch_loss):.5f}")
        dist.barrier()

        evaluation(
            model,
            probes,
            val_dataloader,
            logging,
            global_step,
            epoch,
            postfix,
            world_size,
            device,
        )

        dist.barrier()  # ------------------------ barrier ------------------------------
        if (epoch + 1) % 5 == 0:
            os.makedirs(f"wm-out{postfix}", exist_ok=True)
            torch.save(
                model.state_dict(),
                f"wm-out{postfix}/wm_vqgan_blink_ep{global_step}{postfix}.ckpt",
            )

        epoch += 1

    if dist.get_rank() == 0 and logging:
        wandb.finish()


def eval_step_ar2(
    model,
    probes,
    decoder,
    val_dataloader,
    num_ar_pred,
    emb=None,
    lp_func: LPIPS = None,
    rank=None,
):

    color_palette_torch = torch.tensor(discrete_colors) / 255.0 * 2.0 - 1.0
    color_palette_torch = color_palette_torch.cuda()
    model.eval()
    probes.eval()
    decoder.eval()
    losses = []
    eval_index_acc = []
    if DJEPA:
        codebook = model.module.encoder.quantize.get_codebook_entry
    num_pred = model.module.num_pred
    num_hist = model.module.num_hist
    test_accuracies = []

    lcm = 60
    rep = (num_hist + num_ar_pred) // lcm + 1
    print("rep:", rep)
    print("hist:", num_hist)
    print("pred:", num_ar_pred)
    assert lcm % num_hist == 0
    pick = [0, 10, 100, 999]

    hbar = np.ones((6, (64 + 3) * num_ar_pred, 3), dtype=np.uint8) * 255
    hbar_short = np.ones((6, (64 + 3) * len(pick), 3), dtype=np.uint8) * 255
    vbar_int = torch.ones((num_ar_pred, 64, 3)).long().cuda() * 6
    vbar_short = torch.ones((len(pick), 64, 3)).long().cuda() * 6
    vbar_int_cond = torch.ones((num_hist, 64, 3)).long().cuda() * 6
    skip = 0
    for bid, batch in enumerate(tqdm(val_dataloader)):
        if skip:
            skip -= 1
            continue
        with torch.no_grad():
            batch, _, _ = batch

            B, T, C, H, W = batch["pixel_values"].size()

            # extend batch to infinity -------------------------------------

            # answers (60 period)
            rep_pix = batch["pixel_values_discrete"].cuda().long()  # b t h w
            pixel_values_discrete_extended = rearrange(
                torch.cat((rep_pix[:, num_hist:], rep_pix[:, :num_hist]), dim=1),
                "b t 1 ... -> (b t) ...",
            )
            bg_mask = rearrange(
                (pixel_values_discrete_extended > 0), "(b t) h w -> b t (h w)", b=B
            )
            im1 = color_palette_torch[pixel_values_discrete_extended].permute(
                0, 3, 1, 2
            )
            _valid = (
                pixel_values_discrete_extended > 0
            ) * 1.0  # (b t) h w (with true only colored)
            init_mask_ = _valid[:, None, ...].repeat(1, 3, 1, 1).cuda()  # (b t) 3 h w

            acc_cont = []
            acc_nobg_cont = []
            lpips_cont = []
            mse_cont = []
            conditioning = batch["pixel_values"][:, :num_hist].cuda()

            for i_ in range(rep):
                z_pred = model.module.rollout(
                    {"visual": conditioning},
                    predict_t=lcm,
                    inc=num_hist,
                    output_indices=False,
                    z_dct=(
                        None
                        if i_ == 0
                        else (
                            {"visual_indices": conditioning}
                            if DJEPA
                            else {"visual": conditioning}
                        )
                    ),
                )
                z_pred = z_pred[:, num_hist : num_hist + lcm]  # b t n #v

                if DJEPA:  # output is index
                    # worldmodel was (repr|idx) -> idx
                    z_pred_idx = z_pred[..., 0]
                    conditioning = z_pred_idx[:, -num_hist:]  # last hist (index)
                    if emb:
                        z_pred_quantized = rearrange(
                            emb(z_pred_idx), "b t n d -> (b t) n d"
                        )  # b t n d

                    # r = (loss_components['z_tgt'].detach()==z_pred_idx)
                    # eval_index_acc.append(r.sum(-1)/r.size(-1))  # b t

                    # quantize the predicted indices  b t n -> (b t n) -> (b t n) d -> (b t) n d
                    else:
                        z_pred_quantized = rearrange(
                            torch.nn.functional.normalize(
                                codebook(z_pred_idx.flatten()), p=2, dim=-1
                            ),
                            "(b t n) d -> (b t) n d",
                            b=B,
                            t=lcm,
                        )
                    # z_pred_pooled = torch.flatten(z_pred_quantized, start_dim=1) # (b t) (n d)
                    z_decoder_in = z_pred_quantized

                else:  # output is distributed repr
                    conditioning = z_pred[:, -num_hist:]
                    # eval_index_acc.append(torch.zeros(z_pred.shape[0:2]))  # b t
                    z_pred = rearrange(z_pred, "b t n d -> (b t) n d")
                    # z_pred_pooled = torch.flatten(z_pred, start_dim=1) # (b t) (n d)
                    z_decoder_in = z_pred

                if decoder.init_mask_type == "white":
                    recon = decoder(
                        slots=z_decoder_in, init_mask=init_mask_
                    ).recon  # (b t) 7 h w

                recon = torch.argmax(recon, dim=1)
                # recon = (init_mask_.sum(1)>0).long()*6
                res = rearrange(
                    recon == pixel_values_discrete_extended,
                    "(b t) h w -> b t (h w)",
                    b=B,
                )

                acc_t = res.sum(-1) / res.size(-1)  # b t
                acc_t_nobg = (res * bg_mask).sum(-1) / bg_mask.sum(-1)  # b t

                acc_cont.append(acc_t)
                acc_nobg_cont.append(acc_t_nobg)

                im0 = color_palette_torch[recon].permute(0, 3, 1, 2)
                l = rearrange(lp_func(im0, im1).squeeze(), "(b t) -> b t", b=B)
                m = rearrange(
                    (((im0 - im1) * 0.5) ** 2).flatten(start_dim=1).mean(-1),
                    "(b t) -> b t",
                    b=B,
                )
                lpips_cont.append(l)
                mse_cont.append(m)

            acc_cont = torch.cat(acc_cont, dim=-1)
            acc_nobg_cont = torch.cat(acc_nobg_cont, dim=-1)
            lpips_cont = torch.cat(lpips_cont, dim=-1)  # b t  concat at t
            mse_cont = torch.cat(mse_cont, dim=-1)
            # all  shapes are : batch ar_pred
            # print(acc_cont.shape, acc_nobg_cont.shape, lpips_cont.shape, mse_cont.shape)
            test_accuracies.append(
                torch.stack((acc_cont, acc_nobg_cont, lpips_cont, mse_cont), dim=-1)
            )  # b t 4

            losses.append(0)
            eval_index_acc.append(torch.zeros((B, num_ar_pred)))

            # r = (loss_components['z_tgt'].detach()==z_pred_idx)
            # eval_index_acc.append(r.sum(-1)/r.size(-1))  # b t

            # quantize the predicted indices  b t n -> (b t n) -> (b t n) d -> (b t) n d
            # z_pred_pooled = torch.flatten(z_pred_quantized, start_dim=1) # (b t) (n d)

            # label_last = label[:, num_hist:num_hist+num_pred].cuda()
            # ta = []
            # for i, probe in enumerate(probes):
            #     output = probe(z_pred_pooled)
            #     preds = rearrange(output.argmax(-1),'(b t) -> b t', b=b)
            #     accuracies = (preds == label_last[:, :, i]).float()
            #     t_ta = []
            #     for ts in range(num_pred):
            #         t_ta.append(accuracies[:, ts].mean().item())
            #     ta.append(t_ta)
            # test_accuracies.append(ta)

            # test_accuracies.append(
            #     torch.stack(
            #         (
            #             acc_t,
            #             acc_t_nobg,
            #         ),
            #         dim=-1,
            #     )
            # )  # b t 2

            # recon = recon.view(B, num_ar_pred,3, H, W)
            # gt = pixel_values_discrete_extended.view(B, num_ar_pred,3, H, W)

            # md = "djepa" if DJEPA else "ijepa"
            # with open(f"figs/{md}/rollout_{md}_recon_{bid}.pkl", "wb") as d:
            #     pickle.dump(color_palette[recon.cpu().numpy()].astype(np.uint8), d)
            # with open(f"figs/{md}/rollout_{md}_target_{bid}.pkl", "wb") as d:
            #     pickle.dump(color_palette[gt.cpu().numpy()].astype(np.uint8), d)
            # with open(f"figs/{md}/rollout_{md}_condition_{bid}.pkl", "wb") as d:
            #     pickle.dump(color_palette[cond_discrete.cpu().numpy()].astype(np.uint8), d)

            # for i in range(B):
            #     ex_img = np.concatenate(
            #             (
            #                 color_palette[rearrange(torch.cat((recon[i],vbar_int),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8),
            #                 hbar,
            #                 color_palette[rearrange(torch.cat((gt[i],vbar_int),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8),

            #             ),
            #                 axis=0
            #             )
            #     ex_short_img = np.concatenate(
            #             (
            #                 color_palette[rearrange(torch.cat((recon[i],vbar_short),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8),
            #                 hbar_short,
            #                 color_palette[rearrange(torch.cat((gt[i],vbar_short),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8),

            #             ),
            #                 axis=0
            #             )
            #     cond_img = color_palette[rearrange(torch.cat((cond_discrete[i],vbar_int_cond),dim=-1),'t h w -> h (t w)').cpu().numpy()].astype(np.uint8)
            #     Image.fromarray(
            #         ex_img
            #     ).save(f"figs/{'djepa' if DJEPA else 'ijepa'}/a_wm_{bid:04d}_{i:02d}_rank{rank}.png")
            #     Image.fromarray(
            #         ex_short_img
            #     ).save(f"figs/{'djepa' if DJEPA else 'ijepa'}/a_wm_{bid:04d}_{i:02d}_rank{rank}_{pick}.png")
            #     Image.fromarray(
            #         cond_img
            #     ).save(f"figs/{'djepa' if DJEPA else 'ijepa'}/a_wm_{bid:04d}_{i:02d}_rank{rank}_cond.png")

            del (
                batch,
                pixel_values_discrete_extended,
                recon,
                z_pred,
                rep_pix,
                bg_mask,
                res,
                # z_decoder_in,
            )  # ,image_batch, target_batch,ex_img,ex_short_img,cond_img,
            # if decoder.init_mask_type == "white":
            #    del _valid, init_mask_
            torch.cuda.empty_cache()

    losses = torch.tensor(losses).cuda()
    eval_index_acc = torch.cat(eval_index_acc, dim=0).cuda()
    test_accuracies = torch.cat(test_accuracies, dim=0).cuda()  # [b...b] t 4
    print(test_accuracies.shape)
    return losses, eval_index_acc, test_accuracies


def evaluation_dec_ar(
    model,
    probes,
    decoder,
    val_dataloader,
    postfix,
    num_ar_pred,
    world_size,
    device,
    emb=None,
    rank=None,
):
    _, _, acc_recon = eval_step_ar2(
        model,
        probes,
        decoder,
        val_dataloader,
        num_ar_pred,
        emb=emb,
        lp_func=LPIPS().cuda(),
        rank=rank,
    )
    # print(losses.size(),acc_index.size(),acc_probe.size())
    # local_loss_sizes = torch.tensor(losses.size(), device=device)
    # global_loss_sizes = [torch.zeros_like(local_loss_sizes,device=device) for _ in range(world_size)]

    # dist.all_gather(global_loss_sizes, local_loss_sizes)

    # global_losses = [torch.zeros(tuple(_s),device=device,dtype=losses.dtype) for _s in global_loss_sizes]
    # global_acc_index = [torch.zeros(acc_index.size(),device=device, dtype=acc_index.dtype) for _ in range(world_size)]
    global_acc_recon = [
        torch.zeros(acc_recon.size(), device=device, dtype=acc_recon.dtype)
        for _ in range(world_size)
    ]
    # dist.all_gather(global_losses, losses)
    # dist.all_gather(global_acc_index, acc_index)
    dist.all_gather(global_acc_recon, acc_recon)
    # global_losses = torch.cat(global_losses).cpu()
    # global_acc_index = torch.cat(global_acc_index).cpu()
    global_acc_recon = torch.cat(global_acc_recon).cpu()

    if dist.get_rank() == 0:
        with torch.no_grad():
            # eval_loss = global_losses.mean()
            # eval_acc_index = global_acc_index.mean(dim=0)

            print(global_acc_recon.shape)
            eval_acc_recon = global_acc_recon.mean(dim=0)  # t 2
        # eval_mAP = 0 #MultilabelAveragePrecision(global_labels.shape[1])(nn.functional.sigmoid(global_ypreds), global_labels)
        # eval_acc_top1 = MulticlassAccuracy(global_ypreds.shape[-1])(global_ypreds,global_labels)
        # eval_acc_top5 = MulticlassAccuracy(global_ypreds.shape[-1],top_k=5)(global_ypreds,global_labels)
        # acc = (torch.argmax(global_ypreds,dim=1) == global_labels.long()).sum()/global_labels.numel()
        # print(f"\nEpoch {epoch} val_loss={eval_loss:.5f}, index pred acc: {eval_acc_index.mean()*100}%")
        # log_dict = {}
        for ts, a in enumerate(eval_acc_recon):
            print(
                f"T +{ts+1} recon_acc: {a[0]*100: .4f} %, recon_acc_nobg: {a[1]*100: .4f} %, lpips: {a[2]} %, mse: {a[3]}"
            )  #: index acc: {eval_acc_index[ts]*100:.4f} %')
            # log_dict[f'index_acc_T+{1+ts}']=eval_acc_index[ts]
            # for tid , a_s in enumerate(a):
            #     print(f'\tTask {tid}: {a_s*100:.4f} %',end='\t\t')
            #     log_dict[f'acc_{tid}_T+{1+ts}']= a_s
            # print()

        # log_dict['val_loss']=eval_loss
        # log_dict['index_acc'] =eval_acc_index.mean()
        # if logging:
        # wandb.log({'pixel_acc':acc},step=global_step)
        #    wandb.log(log_dict, step=global_step)

        # hh = 5
        # ww = 8
        # _m = 10
        # _ms = 5
        # imgen = []
        # for i, (im, lab) in enumerate(zip(global_ypreds[:hh*ww],global_labels[:hh*ww])):
        #     new_im = color_palette[im.argmax(0).numpy().astype(int)]
        #     new_lab = color_palette[lab.numpy().astype(int)]
        #     bar = np.ones((new_im.shape[0],_ms,3),dtype=np.uint8)*255
        #     imgen.append(np.concatenate((new_im,bar,new_lab),axis=1))
        #     #a = Image.fromarray(np.concatenate((new_im,new_lab),axis=1),mode='RGB')
        #     #os.makedirs(f'out{postfix}',exist_ok=True)
        #     #a.save(f'out{postfix}/{epoch+1}_{i}.png')

        # allimm = np.ones(((hh+1)*_m + imgen[0].shape[0]*hh, (ww+1)*_m + imgen[0].shape[1]*ww, 3), dtype=np.uint8)*255
        # for i in range(hh):
        #     for j in range(ww):
        #         idx = i*ww+j
        #         if idx >= len(imgen):
        #             break
        #         allimm[_m+i*(_m+imgen[0].shape[0]):_m+i*(_m+imgen[0].shape[0])+imgen[0].shape[0], _m+j*(_m+imgen[0].shape[1]):_m+j*(_m+imgen[0].shape[1])+imgen[0].shape[1]] = imgen[idx]
        # allimm = Image.fromarray(allimm,mode='RGB')
        # os.makedirs(f'out{postfix}',exist_ok=True)
        # allimm.save(f'out{postfix}/{epoch+1}_all.png')


def evaluation(
    model,
    probes,
    val_dataloader,
    logging,
    global_step,
    epoch,
    postfix,
    world_size,
    device,
):
    losses, acc_index, acc_probe = eval_step(model, probes, val_dataloader)
    # print(losses.size(),acc_index.size(),acc_probe.size())
    local_loss_sizes = torch.tensor(losses.size(), device=device)
    global_loss_sizes = [
        torch.zeros_like(local_loss_sizes, device=device) for _ in range(world_size)
    ]

    dist.all_gather(global_loss_sizes, local_loss_sizes)

    global_losses = [
        torch.zeros(tuple(_s), device=device, dtype=losses.dtype)
        for _s in global_loss_sizes
    ]
    global_acc_index = [
        torch.zeros(acc_index.size(), device=device, dtype=acc_index.dtype)
        for _ in range(world_size)
    ]
    global_acc_probe = [
        torch.zeros(acc_probe.size(), device=device, dtype=acc_probe.dtype)
        for _ in range(world_size)
    ]
    dist.all_gather(global_losses, losses)
    dist.all_gather(global_acc_index, acc_index)
    dist.all_gather(global_acc_probe, acc_probe)
    global_losses = torch.cat(global_losses).cpu()
    global_acc_index = torch.cat(global_acc_index).cpu()
    global_acc_probe = torch.cat(global_acc_probe).cpu()
    if dist.get_rank() == 0:

        with torch.no_grad():
            eval_loss = global_losses.mean()
            eval_acc_index = global_acc_index.mean(dim=0)
            eval_acc_probe = global_acc_probe.mean(dim=0)
        # eval_mAP = 0 #MultilabelAveragePrecision(global_labels.shape[1])(nn.functional.sigmoid(global_ypreds), global_labels)
        # eval_acc_top1 = MulticlassAccuracy(global_ypreds.shape[-1])(global_ypreds,global_labels)
        # eval_acc_top5 = MulticlassAccuracy(global_ypreds.shape[-1],top_k=5)(global_ypreds,global_labels)
        # acc = (torch.argmax(global_ypreds,dim=1) == global_labels.long()).sum()/global_labels.numel()
        print(
            f"\nEpoch {epoch} val_loss={eval_loss:.5f}, index pred acc: {eval_acc_index.mean()*100}%"
        )
        log_dict = {}
        for ts, a in enumerate(eval_acc_probe):
            print(f"T +{ts+1} : index acc: {eval_acc_index[ts]*100:.4f} %")
            log_dict[f"index_acc_T+{1+ts}"] = eval_acc_index[ts]
            for tid, a_s in enumerate(a):
                print(f"\tTask {tid}: {a_s*100:.4f} %", end="\t\t")
                log_dict[f"acc_{tid}_T+{1+ts}"] = a_s
            print()

        log_dict["val_loss"] = eval_loss
        log_dict["index_acc"] = eval_acc_index.mean()
        if logging:
            # wandb.log({'pixel_acc':acc},step=global_step)
            wandb.log(log_dict, step=global_step)

        # hh = 5
        # ww = 8
        # _m = 10
        # _ms = 5
        # imgen = []
        # for i, (im, lab) in enumerate(zip(global_ypreds[:hh*ww],global_labels[:hh*ww])):
        #     new_im = color_palette[im.argmax(0).numpy().astype(int)]
        #     new_lab = color_palette[lab.numpy().astype(int)]
        #     bar = np.ones((new_im.shape[0],_ms,3),dtype=np.uint8)*255
        #     imgen.append(np.concatenate((new_im,bar,new_lab),axis=1))
        #     #a = Image.fromarray(np.concatenate((new_im,new_lab),axis=1),mode='RGB')
        #     #os.makedirs(f'out{postfix}',exist_ok=True)
        #     #a.save(f'out{postfix}/{epoch+1}_{i}.png')

        # allimm = np.ones(((hh+1)*_m + imgen[0].shape[0]*hh, (ww+1)*_m + imgen[0].shape[1]*ww, 3), dtype=np.uint8)*255
        # for i in range(hh):
        #     for j in range(ww):
        #         idx = i*ww+j
        #         if idx >= len(imgen):
        #             break
        #         allimm[_m+i*(_m+imgen[0].shape[0]):_m+i*(_m+imgen[0].shape[0])+imgen[0].shape[0], _m+j*(_m+imgen[0].shape[1]):_m+j*(_m+imgen[0].shape[1])+imgen[0].shape[1]] = imgen[idx]
        # allimm = Image.fromarray(allimm,mode='RGB')
        # os.makedirs(f'out{postfix}',exist_ok=True)
        # allimm.save(f'out{postfix}/{epoch+1}_all.png')


def get_postfix(cfg, use_post_vq):

    if DJEPA:
        token_size = str(96)
        num_latent_tokens = str(16)

        vq_mode = f"-V{1024}"

        postfix = f"-T{token_size}-L{num_latent_tokens}{vq_mode}"
    else:
        postfix = ""
    return postfix


def main(cfg, tokenizer, probes, use_post_vq=True):

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
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler,
    )
    # Load Validation Dataset
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=val_sampler,
    )

    # tokenizer.requires_grad_(True)
    # tokenizer.train()

    input_indices = True
    use_post_vq = True
    depth = 2
    heads = 4
    model_dim = 768  # cfg.worldmodel.vq_model.token_size if DJEPA else 768
    mlp_hidden_dim = 4 * model_dim
    attn_head_dim = model_dim // heads
    dropout = 0.3
    emb_dropout = 0.0
    num_hist = 6
    num_pred = num_hist

    if not DJEPA:
        use_post_vq = False
        input_indices = False

    from dino_wm import (
        VWorldModel,
        ViTPredictor,
        ViTCategoricalPredictor,
        ViTIndex2IndexPredictor,
    )

    # I -> I
    predictor = ViTIndex2IndexPredictor(
        num_patches=16,
        num_frames=num_hist,
        codebook_size=1024,
        dim=model_dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_hidden_dim,
        pool="mean",
        dim_head=attn_head_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
    ).cuda()

    model = VWorldModel(
        cfg,
        use_post_vq,
        num_hist=num_hist,
        num_pred=num_pred,
        encoder=tokenizer,
        predictor=predictor,
        train_encoder=False,
        train_predictor=True,
        input_indices=input_indices,
        djepa=DJEPA,
    )
    model = DistributedDataParallel(model.cuda(), find_unused_parameters=False)

    postfix = (
        datetime.now().strftime("%Y%m%d%H%M%S")[2:]
        + get_postfix(cfg, use_post_vq=use_post_vq)
        + f"_{model_dim}dim_{depth}l_{heads}h_H{num_hist}_{'r2r' if not use_post_vq else ('i2i' if input_indices else 'r2i') }"
    )
    if AUTOREGRESSIVE:
        # setting decoder
        from decoders import CrossAttentionDecoder, CrossAttentionDecoderConfig

        decode_indices = False

        decoder_cfg = CrossAttentionDecoderConfig(
            input_size=96,
            hidden_size=64,
            num_hidden_layers=6,
            num_attention_heads=4,
            hidden_dropout_prob=0.0,
            image_size=64,
            patch_size=16,
            num_channels=7,
            num_cls_tokens=1,
            cls_split_to_slots=False,
            init_mask_type="white",
        )
        decoder = CrossAttentionDecoder(decoder_cfg)

        # loading checkpoint
        emb = None

        wm_ckpt = torch.load(
            "",
            map_location="cuda",
        )
        decoder_ckpt = torch.load(
            "",
            map_location="cuda",
        )
        model.load_state_dict(wm_ckpt, strict=True)
        decoder_ckpt_dec = {
            k.removeprefix("module.decoder."): v
            for k, v in decoder_ckpt.items()
            if k.startswith("module.decoder")
        }
        decoder.load_state_dict(decoder_ckpt_dec)
        decoder.cuda()
        # decoder = model.module.encoder.decoder
        # run!
        evaluation_dec_ar(
            model,
            probes,
            decoder,
            val_dataloader,
            postfix,
            AR_PRED,
            world_size,
            device,
            emb=emb,
            rank=rank,
        )
        print("vqgan")
        exit()
    train(cfg, model, probes, train_dataloader, val_dataloader, postfix, logging=True)


def config():
    # Load Configuration
    with initialize(version_base=None, config_path=f"./configs"):
        cfg = compose(config_name="custom_vqgan_blink.yaml")

    cfg.worldmodel.lr = 1e-3

    cfg.data.params.train.params.video_len = 12
    cfg.data.params.validation.params.video_len = 12
    cfg.data.params.train.params.root = "/data/hslee/discrete-jepa/runner/datasets/data/blinking_balls/4Balls_12Frames_Blinking_Pattern"
    cfg.data.params.validation.params.root = "/data/hslee/discrete-jepa/runner/datasets/data/blinking_balls/4Balls_12Frames_Blinking_Pattern"
    cfg.data.params.train.params.stochastic_sample = False
    cfg.data.params.validation.params.stochastic_sample = False

    cfg.worldmodel.total_steps = 1000 * 15  # 15 epochs

    if AUTOREGRESSIVE:
        cfg.data.params.validation.params.data_path = "4Balls_60Frames_Blinking_Pattern"
        cfg.data.params.validation.params.video_len = 60
        cfg.data.params.validation.params.stochastic_sample = False
        cfg.data.params.validation.params.output_discrete_pixel_value = True

    tokenizer_ckpt_path = "/data/hslee/taming-transformers/logs/2025-06-19T15-57-53_custom_vqgan_blink/checkpoints/last.ckpt"
    probe_input_size = 96 * 16

    num_classes_list = [5, 4, 9, 16]

    seed_everything(cfg.seed, workers=True)

    tokenizer = load_tokenizer(
        cfg, tokenizer_ckpt_path, f"cuda:{torch.cuda.current_device()}"
    )

    probes = nn.ModuleList(
        nn.Linear(probe_input_size, num_classes) for num_classes in num_classes_list
    )
    probes.load_state_dict(
        torch.load(
            "cls_blink_lr0.1_ep20000_flatten_vqgan_cat.ckpt",
            map_location="cuda",
        )
    )
    probes = probes.cuda()

    return tokenizer, probes, cfg


if __name__ == "__main__":
    ddp_setup()
    tokenizer, probes, cfg = config()
    main(cfg, tokenizer, probes)
