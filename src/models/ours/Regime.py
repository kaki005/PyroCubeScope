import numpy as np
import numpyro
from numpyro import distributions as dist


class Regime:
    @property
    def A(self):
        return
    @property
    def B(self):
        pass


    def __init__(self, num_topic: int, n_dims, alpha: float, beta:float):
        self.num_topic: int = num_topic
        self.n_dims = n_dims
        self.n_modes: int = len(n_dims)
        """テンソルの階数"""
        self.alpha = np.full(self.n_dims[0], alpha)
        """(トピック数)"""
        self.betas: list[np.ndarray] = [np.full(num_topic, beta) for _ in range(self.n_modes - 1)]
        """(モード数, トピック)"""
        self.factors: list[np.ndarray] = [np.full((d, self.num_topic), 1, dtype=float) for d in self.n_dims]
        """(モード, モードの次元, トピック)"""
        self.counterB = [0 for _ in range(num_topic)]
        """トピックの出現回数"""
        self.counterA = [[0 for j in range(self.n_dims[i])] for i in range(self.n_modes)]
