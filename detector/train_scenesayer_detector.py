#!/usr/bin/env python3

import argparse
import os
import random
from datetime import timedelta
from typing import Dict

import torch
import torch.distributed as dist
import wandb
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ag_dataset_scenesayer import ActionGenomeSceneSayerDetectorDataset, collate_fn
from detector_map import evaluate_detection_map
from dinov3_scenesayer_frcnn import create_scenesayer_detector_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DINOv3 Faster R-CNN detector for SceneSayer")
    parser.add_argument(
        "--data_path",
        default="/home/cse/msr/csy227518/scratch/Datasets/action_genome",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="/home/cse/msr/csy227518/scratch/Projects/Active/scene_sayer_detector/checkpoints",
        type=str,
    )
    parser.add_argument("--backbone_name", default="facebook/dinov3-vitl16-pretrain-lvd1689m", type=str)
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--backbone_lr_ratio", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--box_loss_weight", default=2.0, type=float)
    parser.add_argument("--rpn_box_loss_weight", default=2.0, type=float)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--min_size", default=600, type=int)
    parser.add_argument("--max_size", default=1000, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--datasize", default="full", type=str)
    parser.add_argument("--data_fraction", default=1.0, type=float)
    parser.add_argument("--eval_every", default=1, type=int)
    parser.add_argument("--eval_fraction", default=1.0, type=float)
    parser.add_argument("--eval_max_samples", default=None, type=int)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="scene_sayer_detector", type=str)
    parser.add_argument("--wandb_run_name", default=None, type=str)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.set_defaults(use_amp=torch.cuda.is_available())
    return parser.parse_args()


def reduce_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    reduced = {name: value.detach() for name, value in loss_dict.items()}
    if not dist.is_available() or not dist.is_initialized():
        return {name: float(value.item()) for name, value in reduced.items()}

    with torch.no_grad():
        names = sorted(reduced.keys())
        values = torch.stack([reduced[name] for name in names], dim=0)
        dist.all_reduce(values)
        values /= dist.get_world_size()
    return {name: float(value.item()) for name, value in zip(names, values)}


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(args):
    if not args.distributed:
        return False, 0, 1

    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    args.local_rank = local_rank
    return True, rank, world_size


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def configure_rpn_and_roi(model):
    rpn = getattr(model, "rpn", None)
    if rpn is not None:
        if hasattr(rpn, "pre_nms_top_n_train") and hasattr(rpn, "post_nms_top_n_train"):
            rpn.pre_nms_top_n_train = 2000
            rpn.post_nms_top_n_train = 1000
            rpn.pre_nms_top_n_test = 2000
            rpn.post_nms_top_n_test = 500
        elif hasattr(rpn, "pre_nms_top_n") and isinstance(rpn.pre_nms_top_n, dict):
            rpn.pre_nms_top_n = {"train": 2000, "test": 2000}
            rpn.post_nms_top_n = {"train": 1000, "test": 500}

        if hasattr(rpn, "batch_size_per_image"):
            rpn.batch_size_per_image = 128
        if hasattr(rpn, "positive_fraction"):
            rpn.positive_fraction = 0.5

    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is not None:
        if hasattr(roi_heads, "batch_size_per_image"):
            roi_heads.batch_size_per_image = 256
        if hasattr(roi_heads, "positive_fraction"):
            roi_heads.positive_fraction = 0.5
        if hasattr(roi_heads, "detections_per_img"):
            roi_heads.detections_per_img = 200


def save_checkpoint(path, epoch, model, optimizer, scheduler, args, object_classes):
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scene_sayer_detector": True,
        "detector_family": "torchvision_fasterrcnn",
        "backbone_family": "dinov3",
        "backbone_name": args.backbone_name,
        "freeze_backbone": args.freeze_backbone,
        "backbone_lr_ratio": args.backbone_lr_ratio,
        "box_loss_weight": args.box_loss_weight,
        "rpn_box_loss_weight": args.rpn_box_loss_weight,
        "num_classes": len(object_classes),
        "object_classes": object_classes,
        "transform_config": {
            "min_size": args.min_size,
            "max_size": args.max_size,
        },
    }
    torch.save(checkpoint, path)


def build_optimizer(model, args):
    model_for_optim = model.module if hasattr(model, "module") else model
    backbone_params = []
    head_params = []

    for name, parameter in model_for_optim.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.base_backbone."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    param_groups = []
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": args.lr * args.backbone_lr_ratio,
                "weight_decay": args.weight_decay,
            }
        )

    return torch.optim.AdamW(param_groups)


def get_backbone_lr(optimizer, args) -> float:
    if len(optimizer.param_groups) > 1:
        return float(optimizer.param_groups[1]["lr"])
    return float(optimizer.param_groups[0]["lr"] * args.backbone_lr_ratio)


def main():
    args = parse_args()
    distributed, rank, world_size = setup_distributed(args)
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed + rank)

    if not (0.0 < args.data_fraction <= 1.0):
        raise ValueError("--data_fraction must be in the range (0, 1].")
    if not (0.0 < args.eval_fraction <= 1.0):
        raise ValueError("--eval_fraction must be in the range (0, 1].")

    if torch.cuda.is_available():
        device = torch.device("cuda", args.local_rank if distributed else 0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    if args.use_wandb and is_main_process(rank):
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "data_path": args.data_path,
                "output_dir": args.output_dir,
                "backbone_name": args.backbone_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "lr": args.lr,
                "backbone_lr_ratio": args.backbone_lr_ratio,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "box_loss_weight": args.box_loss_weight,
                "rpn_box_loss_weight": args.rpn_box_loss_weight,
                "freeze_backbone": args.freeze_backbone,
                "min_size": args.min_size,
                "max_size": args.max_size,
                "datasize": args.datasize,
                "data_fraction": args.data_fraction,
                "eval_every": args.eval_every,
                "eval_fraction": args.eval_fraction,
                "eval_max_samples": args.eval_max_samples,
                "distributed": args.distributed,
                "world_size": world_size,
                "seed": args.seed,
                "use_amp": args.use_amp,
            },
        )

    train_dataset = ActionGenomeSceneSayerDetectorDataset(
        data_path=args.data_path,
        phase="train",
        datasize=args.datasize,
    )
    detector_object_classes = train_dataset.object_classes
    val_dataset = ActionGenomeSceneSayerDetectorDataset(
        data_path=args.data_path,
        phase="test",
        datasize=args.datasize,
    )

    if args.data_fraction < 1.0:
        original_train_size = len(train_dataset)
        sample_count = max(1, int(original_train_size * args.data_fraction))
        rng = random.Random(args.seed)
        sampled_indices = rng.sample(range(original_train_size), sample_count)
        train_dataset = Subset(train_dataset, sampled_indices)
        if is_main_process(rank):
            print(
                "using {:.1%} of training data: {} / {} samples".format(
                    args.data_fraction, sample_count, original_train_size
                )
            )

    original_val_size = len(val_dataset)
    eval_target_size = original_val_size
    if args.eval_max_samples is not None:
        eval_target_size = min(eval_target_size, args.eval_max_samples)
    if args.eval_fraction < 1.0:
        eval_target_size = min(eval_target_size, max(1, int(original_val_size * args.eval_fraction)))
    if eval_target_size < original_val_size:
        rng = random.Random(args.seed + 1)
        sampled_indices = rng.sample(range(original_val_size), eval_target_size)
        val_dataset = Subset(val_dataset, sampled_indices)
        if is_main_process(rank):
            print(
                "using {} / {} validation samples".format(
                    eval_target_size, original_val_size
                )
            )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = create_scenesayer_detector_model(
        num_classes=len(detector_object_classes),
        backbone_name=args.backbone_name,
        freeze_backbone=args.freeze_backbone,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    configure_rpn_and_roi(model)
    if distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank] if torch.cuda.is_available() else None,
            output_device=args.local_rank if torch.cuda.is_available() else None,
        )

    optimizer = build_optimizer(model, args)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, int(0.01 * total_steps))
    warmup = LinearLR(optimizer, start_factor=1e-1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.lr * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    scaler = GradScaler(device="cuda", enabled=args.use_amp and torch.cuda.is_available())

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc="epoch {}".format(epoch + 1), disable=not is_main_process(rank))
        for images, targets in progress:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=args.use_amp and torch.cuda.is_available()):
                loss_dict = model(images, targets)
                weighted_loss_dict = {}
                for loss_name, loss_value in loss_dict.items():
                    if loss_name == "loss_box_reg":
                        weighted_loss_dict[loss_name] = loss_value * args.box_loss_weight
                    elif loss_name == "loss_rpn_box_reg":
                        weighted_loss_dict[loss_name] = loss_value * args.rpn_box_loss_weight
                    else:
                        weighted_loss_dict[loss_name] = loss_value
                loss = sum(weighted_loss_dict.values())

            scaler.scale(loss).backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_items = reduce_loss_dict(loss_dict)
            weighted_loss_items = reduce_loss_dict(weighted_loss_dict)
            current_loss = float(loss.detach().item())
            running_loss += float(loss.detach().item())
            progress.set_postfix(
                loss=current_loss,
                cls=loss_items.get("loss_classifier", 0.0),
                box=loss_items.get("loss_box_reg", 0.0),
                lr=optimizer.param_groups[0]["lr"],
            )

            if args.use_wandb and is_main_process(rank):
                step = epoch * len(train_loader) + progress.n
                wandb.log(
                    {
                        "train/loss": current_loss,
                        "train/loss_classifier": loss_items.get("loss_classifier", 0.0),
                        "train/loss_box_reg": loss_items.get("loss_box_reg", 0.0),
                        "train/loss_objectness": loss_items.get("loss_objectness", 0.0),
                        "train/loss_rpn_box_reg": loss_items.get("loss_rpn_box_reg", 0.0),
                        "train/weighted_loss": weighted_loss_items.get("loss_classifier", 0.0)
                        + weighted_loss_items.get("loss_box_reg", 0.0)
                        + weighted_loss_items.get("loss_objectness", 0.0)
                        + weighted_loss_items.get("loss_rpn_box_reg", 0.0),
                        "train/weighted_loss_box_reg": weighted_loss_items.get("loss_box_reg", 0.0),
                        "train/weighted_loss_rpn_box_reg": weighted_loss_items.get("loss_rpn_box_reg", 0.0),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/backbone_lr": get_backbone_lr(optimizer, args),
                        "train/epoch": epoch + 1,
                    },
                    step=step,
                )

        average_train_loss = running_loss / max(1, len(train_loader))
        if is_main_process(rank):
            print("epoch {} train_loss {:.4f}".format(epoch + 1, average_train_loss))
        if args.use_wandb and is_main_process(rank):
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "epoch/train_loss": average_train_loss,
                    "epoch/lr": optimizer.param_groups[0]["lr"],
                },
                step=(epoch + 1) * len(train_loader),
            )

        if (epoch + 1) % args.save_every == 0 and is_main_process(rank):
            checkpoint_path = os.path.join(args.output_dir, "checkpoint_epoch_{:03d}.pth".format(epoch + 1))
            save_checkpoint(
                path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                object_classes=detector_object_classes,
            )
            print("saved {}".format(checkpoint_path))

        if distributed:
            dist.barrier()

        model.eval()
        if (epoch + 1) % args.eval_every == 0 and is_main_process(rank):
            eval_model = model.module if hasattr(model, "module") else model
            metrics = evaluate_detection_map(
                model=eval_model,
                dataloader=val_loader,
                device=device,
                num_classes=len(detector_object_classes),
            )

            if metrics is not None:
                print(
                    "epoch {} mAP {:.4f} AP50 {:.4f} AP75 {:.4f} images {}".format(
                        epoch + 1,
                        metrics["map"],
                        metrics["map_50"],
                        metrics["map_75"],
                        metrics["num_eval_images"],
                    )
                )
                if args.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "eval/map": metrics["map"],
                            "eval/map_50": metrics["map_50"],
                            "eval/map_75": metrics["map_75"],
                            "eval/num_images": metrics["num_eval_images"],
                        },
                        step=(epoch + 1) * len(train_loader),
                    )
        else:
            metrics = None

        if distributed:
            dist.barrier()

        if is_main_process(rank) and metrics is not None:
            print("per_threshold_ap {}".format(metrics["per_threshold_ap"]))

    if is_main_process(rank):
        final_path = os.path.join(args.output_dir, "checkpoint_final.pth")
        save_checkpoint(
            path=final_path,
            epoch=args.epochs - 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            object_classes=detector_object_classes,
        )
        print("saved {}".format(final_path))

    if args.use_wandb and is_main_process(rank):
        wandb.finish()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
