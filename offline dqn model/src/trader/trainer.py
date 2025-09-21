from __future__ import annotations

import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from .buffer import PrioritizedReplayBuffer, ReplayBuffer
from .checkpoint import CheckpointManager
from .config import AppConfig
from .evaluation import evaluate_policy
from .features import DatasetSplit, FeatureScaler, prepare_datasets
from .env import make_environment
from .model import QNetwork, calculate_td_targets
from .utils import ensure_dir, resolve_device, save_scaler, set_global_seed


class Trainer:
    def __init__(self, config: AppConfig, config_path: Path) -> None:
        self.cfg = config
        self.config_path = config_path
        self.device = resolve_device(config.training.device)
        set_global_seed(config.experiment.seed)
        self.datasets: Dict[str, DatasetSplit]
        self.scaler: FeatureScaler
        self.feature_names: list[str]
        self.datasets, self.scaler, self.feature_names = prepare_datasets(config.data)
        self.feature_dim = self.datasets["train"].features.shape[1]
        self.window = self._resolve_window()
        if self.window != self.cfg.data.window:
            print(
                f"Adjusted window from {self.cfg.data.window} to {self.window} to match dataset lengths"
            )
            self.cfg.data.window = self.window
        self.run_dir = self._init_run_directory()
        self.checkpoint_manager = CheckpointManager(self.run_dir / "checkpoints", config.log.top_k_checkpoints)
        self.artifact_dir = ensure_dir(self.run_dir / "artifacts")
        self.metrics_path = self.run_dir / "metrics.csv"
        self.metrics_json_path = self.run_dir / "metrics.jsonl"
        self._init_artifacts()
        self.train_env, self.val_env, self.test_env = self._build_envs()
        self.policy_net, self.target_net, self.optimizer = self._build_models()
        self.scheduler = self._create_scheduler()
        if self.cfg.training.use_per:
            self.replay = PrioritizedReplayBuffer(
                capacity=1_000_000,
                alpha=self.cfg.training.per_alpha,
                beta_start=self.cfg.training.per_beta_start,
                beta_frames=self.cfg.training.per_beta_frames,
            )
        else:
            self.replay = ReplayBuffer(capacity=1_000_000)
        self.global_step = 0
        self.best_metric = -math.inf
        self.no_improve = 0

    def _init_run_directory(self) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_root = Path(self.cfg.experiment.output_dir) / self.cfg.experiment.name
        ensure_dir(run_root)
        run_dir = ensure_dir(run_root / timestamp)
        return run_dir

    def _init_artifacts(self) -> None:
        config_copy_path = self.run_dir / "config_used.yaml"
        shutil.copy2(self.config_path, config_copy_path)
        save_scaler(self.artifact_dir / "feature_scaler.npz", self.scaler.mean, self.scaler.std)
        feature_file = self.artifact_dir / "feature_names.json"
        feature_file.write_text(json.dumps(self.feature_names, indent=2), encoding="utf-8")
        with self.metrics_path.open("w", encoding="utf-8") as f:
            f.write("step,epsilon,loss,lr,val_return,val_sharpe,val_max_dd\n")
        self.metrics_json_path.touch()

    def _resolve_window(self) -> int:
        desired = self.cfg.data.window
        min_len = min(len(split.features) for split in self.datasets.values())
        effective = min(desired, max(2, min_len - 2))
        if effective < 2:
            raise ValueError("Dataset too small for the chosen splits; add more data or reduce window")
        return effective

    def _build_envs(self) -> Tuple:
        window = self.window
        action_levels = self.cfg.environment.action_levels
        cost_bps = self.cfg.environment.cost_bps
        reward_scale = self.cfg.environment.reward_scale
        train_split = self.datasets["train"]
        val_split = self.datasets["val"]
        test_split = self.datasets["test"]
        train_episode_steps = min(len(train_split.features) - window - 1, 4096)
        train_episode_steps = max(train_episode_steps, 128)
        common_kwargs = dict(
            window=window,
            action_levels=action_levels,
            cost_bps=cost_bps,
            reward_scale=reward_scale,
            cooldown_bars=self.cfg.environment.cooldown_bars,
            max_drawdown=self.cfg.environment.max_drawdown,
            trend_filter_abs_return=self.cfg.environment.trend_filter_abs_return,
            volatility_ceiling=self.cfg.environment.volatility_ceiling,
        )
        train_env = make_environment(
            split_features=train_split.features,
            split_prices=train_split.prices,
            random_start=True,
            max_episode_steps=train_episode_steps,
            **common_kwargs,
        )
        val_env = make_environment(
            split_features=val_split.features,
            split_prices=val_split.prices,
            random_start=False,
            max_episode_steps=len(val_split.features) - window - 1,
            **common_kwargs,
        )
        test_env = make_environment(
            split_features=test_split.features,
            split_prices=test_split.prices,
            random_start=False,
            max_episode_steps=len(test_split.features) - window - 1,
            **common_kwargs,
        )
        return train_env, val_env, test_env

    def _build_models(self) -> Tuple[QNetwork, QNetwork, torch.optim.Optimizer]:
        input_dim = self.train_env.observation_size
        num_actions = len(self.cfg.environment.action_levels)
        model_cfg = self.cfg.model
        policy = QNetwork(
            input_dim=input_dim,
            hidden_sizes=model_cfg.hidden_sizes,
            num_actions=num_actions,
            dueling=model_cfg.dueling,
            dropout=model_cfg.dropout,
            use_gru=model_cfg.use_gru,
            window=self.window,
            feature_dim=self.feature_dim,
            gru_hidden_size=model_cfg.gru_hidden_size,
        ).to(self.device)
        target = QNetwork(
            input_dim=input_dim,
            hidden_sizes=model_cfg.hidden_sizes,
            num_actions=num_actions,
            dueling=model_cfg.dueling,
            dropout=model_cfg.dropout,
            use_gru=model_cfg.use_gru,
            window=self.window,
            feature_dim=self.feature_dim,
            gru_hidden_size=model_cfg.gru_hidden_size,
        ).to(self.device)
        target.load_state_dict(policy.state_dict())
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        return policy, target, optimizer

    def _create_scheduler(self):
        cfg = self.cfg.training.lr_scheduler or {"type": "none"}
        scheduler_type = cfg.get("type", "none").lower()
        if scheduler_type == "cosine":
            t_max = int(cfg.get("t_max", self.cfg.training.max_steps))
            eta_min = float(cfg.get("eta_min", 1e-5))
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, t_max),
                eta_min=eta_min,
            )
        if scheduler_type == "step":
            step_size = int(cfg.get("step_size", 50000))
            gamma = float(cfg.get("gamma", 0.5))
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, step_size),
                gamma=gamma,
            )
        return None

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _epsilon(self, step: int) -> float:
        eps_cfg = self.cfg.training.epsilon
        if step < eps_cfg.warmup_steps:
            return eps_cfg.start
        effective_step = step - eps_cfg.warmup_steps
        if eps_cfg.cycle_length and eps_cfg.cycle_length > 0:
            cycle_pos = effective_step % eps_cfg.cycle_length
        else:
            cycle_pos = effective_step
        decay_span = max(1, eps_cfg.decay_steps)
        clamped = min(cycle_pos, decay_span)
        ratio = clamped / decay_span
        value = eps_cfg.start + (eps_cfg.end - eps_cfg.start) * ratio
        return max(min(value, eps_cfg.start), eps_cfg.end)

    def _select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(len(self.cfg.environment.action_levels))
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def _optimize_model(self) -> float | None:
        if len(self.replay) < max(self.cfg.training.batch_size, self.cfg.training.warmup_steps):
            return None
        if self.cfg.training.use_per:
            sample = self.replay.sample(self.cfg.training.batch_size)
            states, actions, rewards, next_states, dones, weights, indices = sample
            weights_t = torch.from_numpy(weights).unsqueeze(1).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.training.batch_size)
            weights_t = None
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
        target_q = calculate_td_targets(rewards_t, dones_t, next_q, self.cfg.training.gamma)
        td_errors = target_q - q_values
        if weights_t is not None:
            loss = (weights_t * F.smooth_l1_loss(q_values, target_q, reduction="none")).mean()
        else:
            loss = F.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.training.gradient_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.cfg.training.use_per:
            new_priorities = td_errors.detach().abs().cpu().numpy().squeeze()
            self.replay.update_priorities(indices, new_priorities)
        return float(loss.item())

    def _log_metrics(self, step: int, epsilon: float, loss: float | None, eval_result) -> None:
        loss_str = f"{loss:.6f}" if loss is not None else ""
        lr = self._current_lr()
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{step},{epsilon:.5f},{loss_str},{lr:.8f},{eval_result.total_return:.6f},{eval_result.sharpe:.6f},{eval_result.max_drawdown:.6f}\n"
            )
        record = {
            "step": step,
            "epsilon": epsilon,
            "loss": loss,
            "lr": lr,
            "total_return": eval_result.total_return,
            "sharpe": eval_result.sharpe,
            "max_drawdown": eval_result.max_drawdown,
        }
        try:
            with self.metrics_json_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(record) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] unable to write json metrics: {exc}", file=sys.stderr)

    def _save_checkpoint(self, name: str, step: int, metric: float | None = None) -> Path:
        state = {
            "step": step,
            "model_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_mean": self.scaler.mean,
            "scaler_std": self.scaler.std,
            "feature_names": self.feature_names,
            "feature_dim": self.feature_dim,
            "window": self.window,
            "config": self.cfg,
            "device": str(self.device),
        }
        return self.checkpoint_manager.save(name, state, metric)

    def train(self) -> dict:
        state = self.train_env.reset()
        loss_value: float | None = None
        for step in trange(1, self.cfg.training.max_steps + 1, desc="training"):
            self.global_step = step
            epsilon = self._epsilon(step)
            action = self._select_action(state, epsilon)
            next_state, reward, done, _ = self.train_env.step(action)
            self.replay.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = self.train_env.reset()
            loss_out = self._optimize_model()
            if loss_out is not None:
                loss_value = loss_out
            if step % self.cfg.training.target_update_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if step % self.cfg.log.print_freq == 0:
                loss_disp = loss_value if loss_value is not None else 0.0
                print(f"step={step} epsilon={epsilon:.4f} loss={loss_disp:.5f}")
            if step % self.cfg.log.eval_interval == 0:
                eval_result = evaluate_policy(self.val_env, self.policy_net, self.device, self.cfg.training.eval_episodes)
                self._log_metrics(step, epsilon, loss_value, eval_result)
                metric_value = getattr(eval_result, "sharpe")
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self.no_improve = 0
                    path = self._save_checkpoint("model_best", step, metric=metric_value)
                    print(f"[BEST] step={step} val_sharpe={metric_value:.3f} saved={path.name}")
                else:
                    self.no_improve += 1
                if self.no_improve >= self.cfg.training.early_stopping.patience:
                    print("Early stopping triggered")
                    break
            if step % self.cfg.log.checkpoint_interval == 0:
                self._save_checkpoint(f"checkpoint_step_{step}", step)
        final_eval = evaluate_policy(self.test_env, self.policy_net, self.device, episodes=1)
        self._log_metrics(self.global_step, self._epsilon(self.global_step), loss_value, final_eval)
        self._save_checkpoint("model_last", self.global_step, metric=final_eval.sharpe)
        test_summary = {
            "total_return": final_eval.total_return,
            "sharpe": final_eval.sharpe,
            "max_drawdown": final_eval.max_drawdown,
        }
        summary_path = self.run_dir / "test_summary.json"
        summary_path.write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
        self.checkpoint_manager.export_metadata(self.run_dir / "checkpoint_index.json")
        return {
            "run_dir": str(self.run_dir),
            "test_result": test_summary,
            "best_metric": self.best_metric,
        }


def run_training(config: AppConfig, config_path: Path) -> dict:
    trainer = Trainer(config, config_path)
    return trainer.train()
