#!/bin/bash

for model in LinearSystem MyPendulum CollisionAvoidance;
do
  for s in {1..5};
  do
    python3 train_SB3.py --model $model --layout 0 --total_steps 1000 --algorithm 'ALL_fast' --seed $s --num_envs 1;
    python3 train_SB3.py --model $model --layout 0 --total_steps 10000 --algorithm 'ALL_fast' --seed $s --num_envs 2;
    python3 train_SB3.py --model $model --layout 0 --total_steps 100000 --algorithm 'ALL_fast' --seed $s --num_envs 10;
    python3 train_SB3.py --model $model --layout 0 --total_steps 1000000 --algorithm 'ALL_fast' --seed $s --num_envs 20;
  done
done