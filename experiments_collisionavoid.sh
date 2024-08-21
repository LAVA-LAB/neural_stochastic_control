#!/bin/bash

all_flags=" --model CollisionAvoidance --expDecr_multiplier 10  --mesh_verify_grid_init 0.002 --mesh_verify_grid_min 0.002 --min_lip_policy_loss 5 --hidden_layers 3 --logger_prefix collision_ablation --silent --ppo_max_policy_lipschitz 10"
p_bounds=(0.9 0.95 0.98)
mesh_loss=(0.001 0.0005 0.0002)
extra_flags1=(" --no-cplip" " --no-improved_softplus_lip" " --no-weighted")
extra_flags2=(" --no-weighted --no-cplip --no-improved_softplus_lip" " --no-local_refinement"  " --no-weighted --no-cplip --no-improved_softplus_lip --no-local_refinement")

for i in {0..2};
do
    for s in {1..5};
    do
        timeout 4000 python3 run.py  --probability_bound ${p_bounds[i]} --mesh_loss ${mesh_loss[i]} --seed $s $all_flags; 
    done
done

for f in "${extra_flags1[@]}";
do
    for i in {0..2};
    do
        for s in {1..5};
        do
            timeout 4000 python3 run.py $f --probability_bound ${p_bounds[i]} --mesh_loss ${mesh_loss[i]} --seed $s $all_flags;
        done
    done
done

for f in "${extra_flags2[@]}";
do
    for i in {0..2};
    do
        for s in {1..5};
        do
            timeout 4000 python3 run.py $f --probability_bound ${p_bounds[i]} --mesh_loss ${mesh_loss[i]} --seed $s $all_flags;
        done
    done
done

# Parse generated results and create tables and figures as presented in the paper
python parse_results.py --folder 'logger/collision_ablation' --plot_x '[0.9,0.95,0.98]';