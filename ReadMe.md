# Running

Probability bound of 99% and 99.9% with L1 norm:

```python3 run.py --model LinearEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --expdecrease_loss_type 2 --probability_bound 0.99 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.00001 --mesh_loss 0.0001 --train_cell_width 0.01 --mesh_verify_grid_init 0.01 --mesh_verify_grid_min 0.01 --plot_intermediate --ppo_load_file 'ckpt/LinearEnv_seed=1_2024-03-04_13-46-22'```
```python3 run.py --model LinearEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --expdecrease_loss_type 2 --probability_bound 0.99 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.00001 --mesh_loss 0.0001 --mesh_train_grid 0.01 --mesh_verify_grid_init 0.01 --mesh_verify_grid_min 0.01 --plot_intermediate  --ppo_load_file 'ckpt/LinearEnv_seed=1_2024-03-04_13-46-22'```
```python3 run.py --model LinearEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --expdecrease_loss_type 2 --probability_bound 0.999 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.00001 --mesh_loss 0.0001 --mesh_train_grid 0.01 --mesh_verify_grid_init 0.01 --mesh_verify_grid_min 0.01 --plot_intermediate```

Same, but with L_infty norm:

```python3 run.py --model LinearEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --expdecrease_loss_type 2 --probability_bound 0.999 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.00001 --mesh_loss 0.00005 --mesh_train_grid 0.005 --mesh_verify_grid_init 0.005 --mesh_verify_grid_min 0.005 --plot_intermediate --linfty```



The main file to run is `run.py`. Command to verify with probability bound 0.99 and a minimum verify mesh of 0.01, and a minimum final mesh of 0.0001, with local refinements:

```python3 run.py --ppo_load_file ckpt/LinearEnv_seed=1_2024-01-05_17-29-25 --model LinearEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --expdecrease_loss_type 2 --probability_bound 0.99 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples```

Pendulum environment:

```python3 run.py --model PendulumEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --expdecrease_loss_type 2 --probability_bound 0.90 --expDecr_multiplier 100 --local_refinement --epochs 100 --perturb_counterexamples --ppo_total_timesteps 1000000 --mesh_refine_min 0.00001 --mesh_loss 0.0001 --mesh_train_grid 0.01 --mesh_loss 0.001 --mesh_verify_grid_init 0.01 --mesh_verify_grid_min 0.01 --plot_intermediate```

Experiment with 3D environment:

```python3 run.py --model Anaesthesia --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 3000 --expdecrease_loss_type 2 --probability_bound 0.8 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_train_grid 0.2 --mesh_verify_grid_init 0.1 --mesh_verify_grid_min 0.1 --mesh_refine_min 0.001 --mesh_loss 0.01 --ppo_total_timesteps 10000000 --ppo_load_file ckpt/Anaesthesia_seed=1_2024-01-24_15-07-12 --plot_intermediate```

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