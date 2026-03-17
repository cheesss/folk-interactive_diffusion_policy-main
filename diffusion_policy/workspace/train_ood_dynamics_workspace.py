"""기존 diffusion_policy workspace 형식을 따라 OOD dynamics 모델을 학습하는 워크스페이스."""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import random

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.ood_utils import (
    encode_policy_obs,
    get_lowdim_keys,
    normalize_action,
)
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainOODDynamicsWorkspace(BaseWorkspace):
    """Frozen reference policy encoder를 재사용해 latent dynamics를 학습한다."""
    include_keys = ["global_step", "epoch", "lowdim_keys"]
    exclude_keys = ("reference_policy", "reference_workspace")

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.global_step = 0
        self.epoch = 0

        self.reference_workspace = None
        self.reference_policy = _load_policy_from_checkpoint(
            cfg.reference_policy.checkpoint,
            cfg.reference_policy.use_ema,
        )
        for param in self.reference_policy.parameters():
            param.requires_grad = False
        self.reference_policy.eval()

        self.lowdim_keys = get_lowdim_keys(cfg.task.shape_meta)
        latent_dim = int(self.reference_policy.obs_encoder.output_shape()[0])
        proprio_dim = int(sum(np.prod(cfg.task.shape_meta["obs"][key]["shape"]) for key in self.lowdim_keys))
        action_dim = int(np.prod(cfg.task.shape_meta["action"]["shape"]))

        OmegaConf.set_struct(self.cfg, False)
        self.cfg.model.latent_dim = latent_dim
        self.cfg.model.proprio_dim = proprio_dim
        self.cfg.model.action_dim = action_dim
        OmegaConf.set_struct(self.cfg, True)

        self.model = hydra.utils.instantiate(self.cfg.model)
        self.optimizer = hydra.utils.instantiate(
            self.cfg.optimizer,
            params=self.model.parameters(),
        )

    def _compute_loss_components(self, batch):
        curr_latent, curr_proprio = encode_policy_obs(
            self.reference_policy,
            batch["obs"],
            self.lowdim_keys,
        )
        next_latent, next_proprio = encode_policy_obs(
            self.reference_policy,
            batch["next_obs"],
            self.lowdim_keys,
        )
        action = normalize_action(self.reference_policy, batch["action"])

        pred_latent, pred_proprio = self.model(curr_latent, curr_proprio, action)
        latent_loss = F.mse_loss(pred_latent, next_latent.detach())
        proprio_loss = F.mse_loss(pred_proprio, next_proprio.detach())
        loss = latent_loss + self.cfg.training.proprio_loss_weight * proprio_loss
        return {
            "loss": loss,
            "latent_loss": latent_loss,
            "proprio_loss": proprio_loss,
        }

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=len(train_dataloader) * cfg.training.num_epochs,
            last_epoch=self.global_step - 1,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.reference_policy.to(device)
        optimizer_to(self.optimizer, device)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                train_loss_values = []
                train_latent_losses = []
                train_proprio_losses = []
                self.model.train()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        loss_dict = self._compute_loss_components(batch)

                        self.optimizer.zero_grad(set_to_none=True)
                        loss_dict["loss"].backward()
                        self.optimizer.step()
                        lr_scheduler.step()

                        loss_cpu = loss_dict["loss"].item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_loss_values.append(loss_cpu)
                        train_latent_losses.append(loss_dict["latent_loss"].item())
                        train_proprio_losses.append(loss_dict["proprio_loss"].item())

                        step_log = {
                            "train_loss": loss_cpu,
                            "train_latent_loss": loss_dict["latent_loss"].item(),
                            "train_proprio_loss": loss_dict["proprio_loss"].item(),
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if cfg.training.max_train_steps is not None and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                step_log = {
                    "train_loss": float(np.mean(train_loss_values)) if train_loss_values else 0.0,
                    "train_latent_loss": float(np.mean(train_latent_losses)) if train_latent_losses else 0.0,
                    "train_proprio_loss": float(np.mean(train_proprio_losses)) if train_proprio_losses else 0.0,
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                }

                if (self.epoch % cfg.training.val_every) == 0:
                    self.model.eval()
                    val_losses = []
                    val_latent_losses = []
                    val_proprio_losses = []
                    with torch.no_grad():
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_dict = self._compute_loss_components(batch)
                                val_losses.append(loss_dict["loss"].item())
                                val_latent_losses.append(loss_dict["latent_loss"].item())
                                val_proprio_losses.append(loss_dict["proprio_loss"].item())
                                if cfg.training.max_val_steps is not None and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                    if val_losses:
                        step_log["val_loss"] = float(np.mean(val_losses))
                        step_log["val_latent_loss"] = float(np.mean(val_latent_losses))
                        step_log["val_proprio_loss"] = float(np.mean(val_proprio_losses))

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                    self.save_checkpoint(tag="latest")

                self.epoch += 1


def _load_policy_from_checkpoint(ckpt_path: str, use_ema: bool):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    if use_ema and getattr(workspace, "ema_model", None) is not None:
        return workspace.ema_model
    return workspace.model
