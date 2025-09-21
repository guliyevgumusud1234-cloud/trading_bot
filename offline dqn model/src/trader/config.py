from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import yaml


@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int = 42


@dataclass
class LogConfig:
    print_freq: int = 1000
    eval_interval: int = 5000
    checkpoint_interval: int = 10000
    top_k_checkpoints: int = 3


@dataclass
class DataSplits:
    train: float
    val: float
    test: float


@dataclass
class DataConfig:
    csv_path: str
    time_col: str | None = None
    price_col: str = "close"
    feature_cols: Sequence[str] = field(default_factory=lambda: ["close"])
    window: int = 128
    splits: DataSplits = field(default_factory=lambda: DataSplits(train=0.7, val=0.15, test=0.15))
    purge_bars: int = 0
    fill_method: str | None = "ffill"


@dataclass
class EnvironmentConfig:
    cost_bps: float = 1.0
    action_levels: Sequence[float] = field(default_factory=lambda: [-1.0, 0.0, 1.0])
    reward_scale: float = 1.0
    cooldown_bars: int = 0
    max_drawdown: float = 1.0
    trend_filter_abs_return: float = 0.0
    volatility_ceiling: float = 0.0


@dataclass
class DistributionalConfig:
    enabled: bool = False
    num_atoms: int = 51
    v_min: float = -0.05
    v_max: float = 0.05


@dataclass
class ModelConfig:
    type: str = "mlp_dueling"
    hidden_sizes: Sequence[int] = field(default_factory=lambda: [256, 256])
    dueling: bool = True
    dropout: float = 0.0
    distributional: DistributionalConfig = field(default_factory=DistributionalConfig)
    use_gru: bool = False
    gru_hidden_size: int = 128


@dataclass
class EpsilonConfig:
    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 50000
    cycle_length: int = 0
    warmup_steps: int = 0


@dataclass
class EarlyStoppingConfig:
    metric: str = "val_sharpe"
    patience: int = 10
    min_delta: float = 0.0


@dataclass
class TrainingConfig:
    device: str = "auto"
    max_steps: int = 100000
    warmup_steps: int = 10000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    epsilon: EpsilonConfig = field(default_factory=EpsilonConfig)
    target_update_interval: int = 4000
    eval_episodes: int = 1
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000
    lr_scheduler: dict = field(default_factory=lambda: {"type": "none"})


@dataclass
class InferenceConfig:
    hysteresis_threshold: float = 0.0
    cooldown_bars: int = 0


@dataclass
class AppConfig:
    experiment: ExperimentConfig
    log: LogConfig
    data: DataConfig
    environment: EnvironmentConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.experiment.output_dir) / self.experiment.name


def _build_app_config(data: dict) -> AppConfig:
    experiment = ExperimentConfig(**data.get("experiment", {}))
    log_cfg = LogConfig(**data.get("log", {}))
    data_cfg = DataConfig(
        csv_path=data["data"]["csv_path"],
        time_col=data["data"].get("time_col"),
        price_col=data["data"].get("price_col", "close"),
        feature_cols=data["data"].get("feature_cols", ["close"]),
        window=data["data"].get("window", 128),
        splits=DataSplits(**data["data"].get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})),
        purge_bars=data["data"].get("purge_bars", 0),
        fill_method=data["data"].get("fill_method", "ffill"),
    )
    env_cfg = EnvironmentConfig(**data.get("environment", {}))
    model_data = data.get("model", {})
    dist_cfg = DistributionalConfig(**model_data.get("distributional", {}))
    model_cfg = ModelConfig(
        type=model_data.get("type", "mlp_dueling"),
        hidden_sizes=model_data.get("hidden_sizes", [256, 256]),
        dueling=model_data.get("dueling", True),
        dropout=model_data.get("dropout", 0.0),
        distributional=dist_cfg,
        use_gru=model_data.get("use_gru", False),
        gru_hidden_size=model_data.get("gru_hidden_size", 128),
    )
    train_data = data.get("training", {})
    epsilon_cfg = EpsilonConfig(**train_data.get("epsilon", {}))
    early_cfg = EarlyStoppingConfig(**train_data.get("early_stopping", {}))
    train_cfg = TrainingConfig(
        device=train_data.get("device", "auto"),
        max_steps=train_data.get("max_steps", 100000),
        warmup_steps=train_data.get("warmup_steps", 10000),
        batch_size=train_data.get("batch_size", 64),
        gamma=train_data.get("gamma", 0.99),
        lr=train_data.get("lr", 1e-3),
        weight_decay=train_data.get("weight_decay", 0.0),
        gradient_clip=train_data.get("gradient_clip", 1.0),
        epsilon=epsilon_cfg,
        target_update_interval=train_data.get("target_update_interval", 4000),
        eval_episodes=train_data.get("eval_episodes", 1),
        early_stopping=early_cfg,
        use_per=train_data.get("use_per", False),
        per_alpha=train_data.get("per_alpha", 0.6),
        per_beta_start=train_data.get("per_beta_start", 0.4),
        per_beta_frames=train_data.get("per_beta_frames", 100000),
        lr_scheduler=train_data.get("lr_scheduler", {"type": "none"}),
    )
    inference_cfg = InferenceConfig(**data.get("inference", {}))
    return AppConfig(
        experiment=experiment,
        log=log_cfg,
        data=data_cfg,
        environment=env_cfg,
        model=model_cfg,
        training=train_cfg,
        inference=inference_cfg,
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _build_app_config(data)
