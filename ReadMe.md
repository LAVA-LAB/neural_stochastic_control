# Running

The main file to run is `run.py`. All important parameters can be set via the command line. For example, a possible run is:

```python3 run.py --ppo_load_file ckpt/LinearEnv_seed=1_2023-12-18_15-23-28 --model LinearEnv --counterx_refresh_fraction 0.5 --epochs 25 --counterx_fraction 0.25 --verify_mesh_tau 0.01 --verify_batch_size 30000```

Which runs the linear model for 25 epochs, using an initial verification mesh size of 0.1, and with 0.25 of the total training data being counterexamples.
After each iteration (consisting of several epochs), the counterexample buffer is refreshed for 50% with new counterexamples.
The batch size `verify_batch_size` indicates that the verifier performs the forward passes over the neural network in batches of 30 thousand (larger batch sizes require too much GPU memory).

# Installing mujoco

- Follow instructions from https://github.com/openai/mujoco-py
- Install GCC 9.x: `brew install gcc@9`

# Installing Box2D

- Requires running `pip3 install wheel setuptools pip --upgrade`

# Running with GPU acceleration

- We tested by installing via Conda with `conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia`.
- Thereafter, install the remaining requirements via pip.

# TODO list

- [ ] Implement sound over/underapproximation of expected decrease condition in the verifier
- [ ] Use less conservative gridding for the verifier. Currently, the grid is "normal", while we ideally want a grid such that any point is at most an epsilon distance away from any vertex (in a weighted L1 norm).
- [ ] Implement a better refinement for the verifier. Currently, the verifier suggests a mesh that is probably needed to verify the conditions, and then checks this mesh across the entire state space. Instead, we can think of a local refinement, because the grid refinement may only be needed in a few places of the state space.
- [ ] Investigate if we can improve the verifier using results from IA-AI.
- [ ] Tidy up the impplementation/updating of the counterexample buffer.