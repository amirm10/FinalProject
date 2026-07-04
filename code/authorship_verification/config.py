"""
Central configuration for the Authorship Verification Framework.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class PathConfig:
    root_dir: str = "."
    data_dir: str = "data_raw"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    logs_dir: str = "logs"
    def __post_init__(self):
        for d in [self.data_dir, self.checkpoint_dir, self.results_dir, self.logs_dir]:
            os.makedirs(os.path.join(self.root_dir, d), exist_ok=True)


@dataclass
class DataConfig:
    bert_model_name: str = "bert-base-uncased"
    max_chunk_tokens: int = 512
    min_chunks_per_text: int = 1
    impostor_ratio: float = 1.0
    num_impostors_per_author: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    dataset_source: str = "pan2020"


@dataclass
class ModelConfig:
    bert_model_name: str = "bert-base-uncased"
    bert_hidden_size: int = 768
    freeze_bert_layers: int = 6
    cnn_num_filters: int = 128
    cnn_kernel_sizes: tuple = (3, 4, 5)
    cnn_dropout: float = 0.3
    bilstm_hidden_size: int = 128
    bilstm_num_layers: int = 2
    bilstm_dropout: float = 0.3
    embedding_dim: int = 256
    fc_hidden_dim: int = 128
    fc_dropout: float = 0.4


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    warmup_steps: int = 500
    scheduler_type: str = "linear"
    batch_size: int = 4         # 8/16 caused CUDA OOM on T4 with multi-chunk texts
    num_epochs: int = 10
    gradient_accumulation_steps: int = 8  # effective batch = 4 * 8 = 32
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.005
    save_every_n_epochs: int = 2
    keep_top_k_checkpoints: int = 3
    contrastive_margin: float = 1.5
    seed: int = 42


@dataclass
class PipelineConfig:
    dtw_window_size: Optional[int] = None
    isolation_forest_contamination: float = 0.45
    isolation_forest_n_estimators: int = 200
    isolation_forest_random_state: int = 42
    kmedoids_n_clusters: int = 2
    kmedoids_random_state: int = 42


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


cfg = Config()
