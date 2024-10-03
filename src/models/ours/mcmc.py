import jax.numpy as jnp
import numpy as np
import numpyro as pyro
from jax import random, vmap
from jax.scipy.special import logsumexp
from numpyro import distributions as dist


def generate(A_ini:jnp.ndarray, B_ini: jnp.ndarray, n_modes:int):        # トピック
    B = pyro.param("B", B_ini)
    topic = pyro.sample(
        "topic_weights", dist.Multinomial(1, B)
    )
    # イベント
    with pyro.plate("index", n_modes) as i:
        A_i = pyro.param(f"A_{i}", A_ini[i][topic])
        event_index = pyro.sample(
            f"index", dist.Multinomial(1, A_i[topic])
        )
    return topic, event_index

def gibbs_sampling(self, rng_key, gibbs_sites, hmc_sites):
    event_index = hmc_sites["index"]
    topic = hmc_sites["topic"]
    B =
