python3 run.py --model Anaesthesia --counterx_refresh_fraction 0.5 --counterx_fraction 0.25 --verify_batch_size 10000 --expdecrease_loss_type 2 --probability_bound 0.9 --expDecr_multiplier 100 --local_refinement --epochs 3 --perturb_counterexamples --mesh_refine_min 0.00001 --mesh_loss 0.005 --mesh_train_grid 0.05 --mesh_verify_grid_init 0.05 --mesh_verify_grid_min 0.05 --plot_intermediate --max_refine_factor 4 --noise_partition_cells 6
