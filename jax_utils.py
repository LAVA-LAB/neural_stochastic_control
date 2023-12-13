import jax.numpy as jnp
from flax.training import train_state
import optax
import jax

vsplit_fun = jax.vmap(jax.random.split)
def vsplit(keys):
    return vsplit_fun(keys)

def create_train_state(model, rng, in_dim, learning_rate=0.01, ema=0):

    params = model.init(rng, jnp.ones([1, in_dim]))
    tx = optax.adam(learning_rate)
    if ema > 0:
        tx = optax.chain(tx, optax.ema(ema))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def lipschitz_coeff_l1(params):
    # Initialize
    L = 1

    # Compute Lipschitz coefficient by iterating through layers
    for layer in params["params"].values():

        L *= jnp.max(jnp.sum(jnp.abs(layer["kernel"]), axis=0))

    return L