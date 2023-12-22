import jax.numpy as jnp
import optax
import jax
from functools import partial
from flax.training.train_state import TrainState
from typing import Callable
from flax import struct

vsplit_fun = jax.vmap(jax.random.split)
def vsplit(keys):
    return vsplit_fun(keys)

def apply_ibp_rectangular(act_fns, params, mean, radius):
    '''
    Implementation of the interval bound propagation (IBP) method from https://arxiv.org/abs/1810.12715.
    We use IBP to compute upper and lower bounds for (hyper)rectangular input sets.

    This function returns the same result as jax_verify.interval_bound_propagation(apply_fn, initial_bounds). However,
    the jax_verify version is generally slower, because it is written to handle more general neural networks.

    :param act_fns: List of flax.nn activation functions.
    :param params: Parameter dictionary of the network.
    :param mean: 2d array, with each row being an input point of dimension n.
    :param radius: 1d array, specifying the radius of the input in every dimension.
    :return: lb and ub (both 2d arrays of the same shape as `mean`
    '''

    # Broadcast radius to match shape of the mean numpy array
    radius = jnp.broadcast_to(radius, mean.shape)

    # Enumerate over the layers of the network
    for i,act_fn in enumerate(act_fns):
        layer = 'Dense_'+str(i)

        # Compute mean and radius after the current fully connected layer
        mean = mean @ params['params'][layer]['kernel'] + params['params'][layer]['bias']
        radius = radius @ jnp.abs(params['params'][layer]['kernel'])

        # Then, apply the activation function and determine the lower and upper bounds
        lb = act_fn(mean - radius)
        ub = act_fn(mean + radius)

        # Use these upper bounds to determine the mean and radius after the layer
        mean = (ub + lb) / 2
        radius = (ub - lb) / 2

    return lb, ub

class AgentState(TrainState):
    # Setting default values for agent functions to make TrainState work in jitted function
    ibp_fn: Callable = struct.field(pytree_node=False)

def create_train_state(model, act_funcs, rng, in_dim, learning_rate=0.01, ema=0, params=None):

    if params is None:
        params = model.init(rng, jnp.ones([1, in_dim]))
    else:
        params = params

    tx = optax.adam(learning_rate)
    if ema > 0:
        tx = optax.chain(tx, optax.ema(ema))
    return AgentState.create(apply_fn=jax.jit(model.apply), params=params, tx=tx,
                             ibp_fn=jax.jit(partial(apply_ibp_rectangular, act_funcs)))

@partial(jax.jit, static_argnums=(1,2,))
def lipschitz_coeff_l1(params, weights=True, CPLip=True):
    if (not weights and not CPLip):
        L = jnp.float32(1)
        # Compute Lipschitz coefficient by iterating through layers
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                L *= jnp.max(jnp.sum(jnp.abs(layer["kernel"]), axis=1))

    elif (not weights and CPLip):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        nmatrices = len(matrices)
        products = [matrices]
        prodnorms = [[jnp.max(jnp.sum(jnp.abs(mat), axis=0)) for mat in matrices]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.sum(jnp.abs(mat), axis=0)) for mat in prod_list])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # interpret idx as binary number of length nmatrices - 1,
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0:  # last one always true
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs


    elif (weights and not CPLip):
        L = jnp.float32(1)
        weights = [jnp.ones(1)]
        # Compute Lipschitz coefficient by iterating through layers
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                colsums = jnp.sum(jnp.multiply(jnp.abs(layer["kernel"]), weights[-1][:, jnp.newaxis]), axis=0)
                lip = jnp.max(colsums)
                weights.append(colsums / lip)
                L *= lip

    elif (weights and CPLip):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        weights = [jnp.ones(1)]
        # Compute Lipschitz coefficient by iterating through layers
        for mat in matrices:
            colsums = jnp.sum(jnp.multiply(jnp.abs(mat), weights[-1][:, jnp.newaxis]), axis=0)
            lip = jnp.max(colsums)
            weights.append(colsums / lip)

        nmatrices = len(matrices)
        products = [matrices]
        extra0 = []
        prodnorms = [[jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(matrices[idx]),
                                                                weights[idx][:, jnp.newaxis]), axis=0),
                                           jnp.float32(1) / weights[idx + 1]))
                      for idx in range(nmatrices)]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(prod_list[idx]),
                                                                        weights[idx][:, jnp.newaxis]), axis=0),
                                                   jnp.float32(1) / weights[idx + nprods + 1]))
                              for idx in range(nmatrices - nprods)])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # interpret idx as binary number of length nmatrices - 1,
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0:  # last one always true
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs

    return L
