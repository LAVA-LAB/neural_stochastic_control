python3 run.py --model LinearEnv --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 30000 --expdecrease_loss_type 2 --probability_bound 0.999 --expDecr_multiplier 100 --local_refinement --epochs 25 --perturb_counterexamples --mesh_refine_min 0.00001 --mesh_loss 0.0001 --mesh_train_grid 0.01 --mesh_verify_grid_init 0.01 --mesh_verify_grid_min 0.01 --plot_intermediate