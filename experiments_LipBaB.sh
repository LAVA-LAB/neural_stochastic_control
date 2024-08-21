#!/bin/bash

for f in "linsys_twolayers_0point99" "linsys_twolayers_0point99995" "linsys_threelayers_0point99" "linsys_threelayers_0point99995";
do
    for s in {1..5};
    do
        echo $f $s;
        checkpoint="ckpt_lipbab/${f}_seed${s}/final_ckpt";
        python3 LipBaB_finalcheckpoint.py --checkpoint $checkpoint;
    done
done
