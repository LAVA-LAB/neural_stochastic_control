#!/bin/bash

all_flags=" --model LinearSystem --expDecr_multiplier 10 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --min_lip_policy_loss 0.5 --hidden_layers 3 --logger_prefix linsys_ablation --silent"
p_bounds=(0.99 0.995 0.999 0.9995 0.9999 0.99995)
mesh_loss=(0.0001 0.0001 0.0001 0.00005 0.00001 0.000005)
mesh_loss2=(0.001 0.0005 0.0001)
extra_flags1=(" --no-weighted" " --no-cplip" " --no-improved_softplus_lip")
extra_flags2=(" --no-weighted --no-cplip --no-improved_softplus_lip")
extra_flags3=(" --no-local_refinement" " --no-weighted --no-cplip --no-improved_softplus_lip --no-local_refinement")

for i in {0..5};
do
    for s in {1..5};
    do
        timeout 4000 python3 run.py  --probability_bound ${p_bounds[i]} --mesh_loss ${mesh_loss[i]} --seed $s $all_flags; 
    done
done

for f in "${extra_flags1[@]}";
do
    for i in {0..5};
    do
        for s in {1..5};
        do
            timeout 4000 python3 run.py $f --probability_bound ${p_bounds[i]} --mesh_loss ${mesh_loss[i]} --seed $s $all_flags;
        done
    done
done

for f in "${extra_flags2[@]}";
do
    for i in {0..3};
    do
        for s in {1..5};
        do
            timeout 4000 python3 run.py $f --probability_bound ${p_bounds[i]} --mesh_loss ${mesh_loss[i]} --seed $s $all_flags;
        done
    done
done

for f in "${extra_flags3[@]}";
do
    for i in {0..2};
    do
        for s in {1..5};
        do
            timeout 4000 python3 run.py $f --probability_bound ${p_bounds[i]} --mesh_loss ${mesh_loss2[i]} --seed $s $all_flags;
        done
    done
done

# Parse generated results and create tables and figures as presented in the paper
python parse_results.py --folder 'logger/linsys_ablation' --plot_x '[0.995,0.9995,0.99995]';