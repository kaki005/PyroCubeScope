import argparse
import os
import time

import gpjax as gpx
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from config import ModelConfig
from numpyro import distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from pydoc_data import topics
from scipy.special import expit


class TopicGP:
    def __init__(self, output_dim: int, latent_dim:int, rand_key: int, config: ModelConfig):
        #TODO: BayesNewtonを使う。
        self.out_dim = output_dim
        self.latent_dim = latent_dim
        self.S = jnp.diag(jnp.abs(random.normal(rand_key, shape=(latent_dim,))))
        """eigenvalue"""
        self.U = random.orthogonal(rand_key, output_dim)[:latent_dim]
        """eigen vector"""
        self.D:jnp.ndarray = jnp.eye(latent_dim)* config.D_init
        self.sigma_2: float = config.sigma_init


    @property
    def W(self):
        """mixing matrix"""
        return self.U @ self.S
    @property
    def Sigma(self):
        """observation noise mat"""
        return self.W @ self.D @ self.W + self.sigma_2 *jnp.eye(self.out_dim)
    @property
    def T(self):
        """projection into latent space"""
        return jnp.sqrt(self.S) @ self.U

    def predict(self, t:float):
        pass
    def update(self, y:jnp.ndarray):
        latent_val = self.T @ y

        pass



class Regime:
    def __init__(self, num_topic: int, n_dims, config:ModelConfig):
        self.num_topic: int = num_topic
        self.key = random.key(config.seed)
        self.n_dims = n_dims
        self.mode_num: int = len(n_dims)
        """num of modes"""
        subkey, _ = random.split(self.key)
        self.B: TopicGP = TopicGP(num_topic, 3, subkey, config)
        """dynamics of topic"""
        self.A :list[list[TopicGP]] = []
        """mode dynamics (mode, topic)"""
        for i in range(self.mode_num):
            self.A.append([])
            for k in range(num_topic):
                subkey, _ = random.split(self.key)
                self.A[i].append(TopicGP(self.n_dims[i], config.latent_dims[i], subkey, config))

    def predict(self, t:float):
        pass


    def update_online(self, t:float, obs_events:jnp.ndarray):
        pass
