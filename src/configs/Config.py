from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ModelConfig:
    k: int = 4
    """topic num"""
    mode_num: int = 2
    """num of mode"""
    latent_ndims= (2,2)
    seed: int = 4
    """seed for jax.random"""
    D_init :float = 0.2
    sigma_init: float = 0.1
    """init of obs noise"""






@dataclass_json
@dataclass
class WandbConfig:
    entity: str = ""
    project: str = ""


@dataclass_json
@dataclass
class Config:
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data_dir: str = ""
    output_dir: str = ""
