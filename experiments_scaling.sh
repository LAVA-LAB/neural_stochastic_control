#!/bin/bash

all_flags="--model LinearSystem --expDecr_multiplier 10 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --min_lip_policy_loss 1 --hidden_layers 3 --layout 1 --logger_prefix linsys_layout1 --silent"
p_bounds=(0.8 0.9 0.95)
mesh_loss=(0.0002 0.0001 0.00005)

for i in {0..2};
do
    for s in {1..5};
    do
        timeout 4000 python3 run.py  --probability_bound ${p_bounds[i]} --mesh_loss ${mesh_loss[i]} --seed $s $all_flags;
    done
done

# Parse generated results and create tables and figures as presented in the paper
python parse_results.py --folder 'logger/linsys_layout1' --plot_x '[0.8,0.9,0.95]';

all_flags=" --model TripleIntegrator --expDecr_multiplier 10 --mesh_verify_grid_init 0.04 --mesh_verify_grid_min 0.04  --hidden_layers 3 --noise_partition_cells 6 --max_refine_factor 4 --verify_batch_size 20000 --min_fraction_samples_per_region 0.1  --logger_prefix tripleintegrator --silent"
p_bounds=(0.8 0.9 0.95)
mesh_loss=(0.002 0.002 0.001)

for i in {0..2};
do
    for s in {1..5};
    do
        timeout 4000 python3 run.py  --probability_bound ${p_bounds[i]} --mesh_loss ${mesh_loss[i]} --seed $s $all_flags; 
    done
done

# Parse generated results and create tables and figures as presented in the paper
python parse_results.py --folder 'logger/tripleintegrator' --plot_x '[0.8,0.9,0.95]';
