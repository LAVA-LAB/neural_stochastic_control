import jax
import jax.numpy as jnp

n = 100000

key = jax.random.PRNGKey(1)
keys = jax.random.split(key, n)

def fun(x, key):
    y = jax.random.normal(key) + x
    new_key, _ = jax.random.split(key)
    return y, new_key

vfun = jax.vmap(fun, in_axes=(0,0), out_axes=0)
y, new_key = vfun(jnp.zeros(n), keys)

print(y.mean())