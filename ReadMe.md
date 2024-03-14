# Pretraining with different RL algorithms

By default, a JAX implementation of PPO is used to pretrain the policy. This corresponds with the option

```--pretrain_method PPO_JAX```

Alternatively, stable-baselines3 implementations of several RL algorithms can be used, namely:
- PPO
- TD3
- SAC
- A2C
- DDPG
To use these options, replace `PPO_JAX` above with one of these algorithms, e.g., `--pretrain_method SAC`.

The other arguments relevant for the pretraining are:
- `--load_ckpt` (string) - Loads the checkpoint, instead of training a new policy (checkpoint must be saved using Orbax).
- `--pretrain_total_steps` (int) - Total number of steps to train the policy for (1 million by default, but this may take very long when using stable-baselines).
- `--pretrain_num_envs` (int) - Number of environments to use in parallel for training (10 by default).

# Installing mujoco

- Follow instructions from https://github.com/openai/mujoco-py
- Install GCC 9.x: `brew install gcc@9`

# Installing Box2D

- Requires running `pip3 install wheel setuptools pip --upgrade`

# Running with GPU acceleration

- We tested by installing via Conda with `conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia`.
- Thereafter, install the remaining requirements via pip (except for jax and jaxlib, because they are already installed via Conda!).

# TODO list

- [ ] Implement sound over/underapproximation of expected decrease condition in the verifier
- [ ] Use less conservative gridding for the verifier. Currently, the grid is "normal", while we ideally want a grid such that any point is at most an epsilon distance away from any vertex (in a weighted L1 norm).
- [ ] Implement a better refinement for the verifier. Currently, the verifier suggests a mesh that is probably needed to verify the conditions, and then checks this mesh across the entire state space. Instead, we can think of a local refinement, because the grid refinement may only be needed in a few places of the state space.
- [ ] Investigate if we can improve the verifier using results from IA-AI.
- [ ] Tidy up the impplementation/updating of the counterexample buffer.