from dataclasses import dataclass


@dataclass
class SamplingConfig:
    sigma_1: float = 0.02
    t_min: float = 1e-10


@dataclass
class TrainingConfig:
    steps: int = 50000
    lr: float = 1e-3
    batch_size: int = 10
