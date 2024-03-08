# Linear Env (2D)
# L1, p=0.9995 (should finish in around 6 minutes)
python3 run.py --model LinearEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --probability_bound 0.9995 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.000001 --mesh_loss 0.00005 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --ppo_load_file 'ckpt/LinearEnv_seed=1_2024-03-08_14-35-02' --new_cx_buffer

# L1, p=0.999 (should finish in around 4 minutes)
python3 run.py --model LinearEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --probability_bound 0.999 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.0000001 --mesh_loss 0.0001 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --plot_intermediate --ppo_load_file 'ckpt/LinearEnv_seed=1_2024-03-06_10-48-38'

# L1, p=0.9
python3 run.py --model LinearEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --probability_bound 0.9 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.0000001 --mesh_loss 0.0001 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --plot_intermediate --debug_train_step --ppo_load_file 'ckpt/LinearEnv_seed=1_2024-03-08_14-35-02' --new_cx_buffer

# Pendulum
# L1
python3 run.py --model PendulumEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --probability_bound 0.9 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.000001 --mesh_loss 0.0001 --mesh_verify_grid_init 0.01 --mesh_verify_grid_min 0.01 --plot_intermediate --no-split_lip
python3 run.py --model PendulumEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --probability_bound 0.99 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.000001 --mesh_loss 0.0001 --mesh_verify_grid_init 0.01 --mesh_verify_grid_min 0.01 --plot_intermediate --no-split_lip
python3 run.py --model PendulumEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --probability_bound 0.99 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.000001 --mesh_loss 0.0001 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --plot_intermediate --no-split_lip


##########
### NOTE: BELOW ARE COMMANDS COMPATIBLE WITH THE OLD "MAIN" BRANCH!
##########

# Dubins car
# L1
python3 run.py --model Dubins --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --probability_bound 0.5 --expDecr_multiplier 100 --local_refinement --epochs 3 --perturb_counterexamples --mesh_refine_min 0.000001 --mesh_loss 0.0001 --mesh_train_grid 0.01 --mesh_verify_grid_init 0.01 --mesh_verify_grid_min 0.01 --max_refine_factor 2 --plot_intermediate --ppo_total_timesteps 20000000 --ppo_max_policy_lipschitz 20 --ppo_load_file 'ckpt/Dubins_seed=1_2024-02-29_08-47-26'
