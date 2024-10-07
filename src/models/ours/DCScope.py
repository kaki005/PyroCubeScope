import numpyro
from config import Config
from numpyro import distributions as dist

from .Regime import Regime


class DCScope:
    def __init__(self, num_topic: int, config: Config, n_dims):
        self.regimes: list[Regime] = []
        self.regimes.append(Regime(num_topic, n_dims, config.Model))
        self.regime_history :list[int] = []
        self.prev_regime: int = 0
        """前回のリジーム"""
    def infer_online(self, tensor):
         pass
