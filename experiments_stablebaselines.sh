#!/bin/bash

steps=(10000 100000 1000000)
algos=("TRPO" "SAC" "TQC" "A2C")

all_flags=" --model LinearSystem --expDecr_multiplier 10 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --min_lip_policy_loss 0.5 --hidden_layers 3 --logger_prefix linsys_sb --silent"
for i in {0..2};
do
  for j in {0..3};
  do
    for s in {1..5};
    do
        checkpoint="ckpt_pretrain_sb3/LinearSystem_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${s}_steps=${steps[i]}"
        timeout 1850 python3 run.py --load_ckpt $checkpoint --probability_bound 0.999 --mesh_loss 0.0001 --seed $s $all_flags;
    done
  done
done

# Parse generated results and create tables and figures as presented in the paper
python parse_results.py --folder 'logger/linsys_sb' --sb --plot_x '[10000,100000,1000000]';

all_flags=" --model MyPendulum --expDecr_multiplier 10  --mesh_verify_grid_init 0.0014 --mesh_verify_grid_min 0.0014 --min_lip_policy_loss 0.1 --hidden_layers 3 --logger_prefix pendulum_sb --silent"
for i in {0..2};
do
  for j in {0..3};
  do
    for s in {1..5};
    do
        checkpoint="ckpt_pretrain_sb3/MyPendulum_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${s}_steps=${steps[i]}"
        timeout 1850 python3 run.py --load_ckpt $checkpoint --probability_bound 0.9 --mesh_loss 0.0001 --seed $s $all_flags;
    done
  done
done

# Parse generated results and create tables and figures as presented in the paper
python parse_results.py --folder 'logger/pendulum_sb' --sb --plot_x '[10000,100000,1000000]';

all_flags="--model CollisionAvoidance --expDecr_multiplier 10 --mesh_verify_grid_init 0.002 --mesh_verify_grid_min 0.002 --min_lip_policy_loss 5 --hidden_layers 3 --logger_prefix collision_sb --silent"
for i in {0..2};
do
  for j in {0..3};
  do
    for s in {1..5};
    do
        checkpoint="ckpt_pretrain_sb3/CollisionAvoidance_layout=0_alg=${algos[j]}_layers=3_neurons=128_outfn=None_seed=${s}_steps=${steps[i]}"
        timeout 1850 python3 run.py --load_ckpt $checkpoint --probability_bound 0.9 --mesh_loss 0.001 --seed $s $all_flags;
    done
  done
done

# Parse generated results and create tables and figures as presented in the paper
python parse_results.py --folder 'logger/collision_sb' --sb --plot_x '[10000,100000,1000000]';