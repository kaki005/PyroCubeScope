import numpyro
from numpyro import distributions as dist
from .Regime import Regime


        """function for gibbs sampling

        Args:
            rng_key (_type_): _description_
            gibbs_sites (_type_): _description_
            hmc_sites (_type_): _description_
        """
        self.counterB[topic]+= 1
        for i, index in enumerate(event_index):
            self.counterA[i][index]+= 1
        pass




class CubeScope:
    def __init__(self, num_topic: int, n_dims: torch.Shape, alpha: float, beta:float):
        self.regimes: list[Regime] = []
        self.regimes.append(Regime(num_topic, n_dims, alpha, beta))
        self.regime_history :list[int] = []
        self.prev_regime: int = 0
        """前回のリジーム"""

    def infer_online(self, tensor):
