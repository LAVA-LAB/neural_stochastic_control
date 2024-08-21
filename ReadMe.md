Learning-Based Verification of Stochastic Dynamical Systems with Neural Network Policies
=============================

This repository contains the supplementary code for the paper:

- [1] "Learning-Based Verification of Stochastic Dynamical Systems with Neural Network Policies." Anonymous submission.

This paper proposes techniques that make the verification of neural network policies in stochastic dynamical systems
more scalable.
In this artifact, we implement these techniques in a learner-verifier framework for verifying that a given neural
network policy satisfies a given reach-avoid specification.
The learner trains another neural network, which acts as a certificate proving that the policy satisfies the task.
The verifier then checks whether this neural network certificate is a so-called reach-avoid supermartingale (RASM),
which suffices to show reach-avoid guarantees.
For more details about the approach, we refer to the main paper.

## Reproducibility

All experiments presented in [1] are run on a server running Ubuntu 22.04.1 LTS, with an Intel Core i9-10980XE CPU, 256
GB of RAM, and an NVIDIA RTX 3090 GPU.
The excepted run times provided in this ReadMe are also considering a server with these specifications.

<br />

# 1. What does this code do?

While we refer to the paper [1] for details, we briefly explain what our code computes.
In a nutshell, given

1) a stochastic dynamical system,
2) a neural network policy, and
3) a reach-avoid specification, i.e., a tuple $(X_T, X_U, \rho)$ of a set of target states $X_T \subset X$, a set of
   unsafe states $X_U \subset X$, and a probability bound $\rho \in (0,1)$.

we compute whether the policy, when deployed on this system, satisfies the reach-avoid specification.
More precisely, a policy $\pi$ satisfies the specification if, from every state in a set $X_0 \subset X$ of initial
states, the probability to reach $X_T$ while never reaching $X_U$ is at least $\rho$.

Our algorithm verifies that the reach-avoid specification is satisfied by learning a formal certificate, called a
reach-avoid supermartingale (RASM), in the form of a neural network.
Finding a RASM is a sufficient proof for the satisfaction of the specification.
Our code implements an iterative learner-verifier framework that tries to find a RASM for the given inputs.
If a valid RASM is found, our code terminates and returns the RASM as proof that the specification is satisfied.

<br />

# 2. Installing

We recommend installing in a conda environment; however, other ways of installing are also possible.
Below, we list the steps needed to install via (Mini)conda.

## Step 1: Install Miniconda

Download Miniconda, e.g., using the following commands ([see here](https://docs.anaconda.com/free/miniconda/) for
details):

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Then initialize Miniconda using the following commands (and close and re-open the terminal):

```
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

## Step 2: Create a new conda environment

The following command creates a new conda environment, specifically with Python version 3.12:

```
conda create -n neural python=3.12
conda activate neural
```

Next, proceed with either step 3a or 3b (for installing with or without GPU acceleration via CUDA, respectively).

## Step 3a: Installing with GPU acceleration

Our implementation uses Jax, which, in turn, can use GPU acceleration via CUDA.
However, installing CUDA can be tricky in some cases.
Thus, we recommend installing via Conda with

```
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

Then, install the remaining requirements using pip:

```
pip3 install -r requirements_no_jax.txt
```

It is important to run this requirements script (and not `requirements.txt`), because jax and and jaxlib are already
installed in the previous `conda install` command.

## Step 3b: Installing without GPU acceleration

If you wish to run without GPU acceleration (or do not have a CUDA-compatible GPU), install all requirements using pip
as follows:

```
pip3 install -r requirements.txt
```

<br />

# 3. Running for a single benchmark

The main Python file to run the code is `run.py`.
See [Section 6](#6-overview-of-input-arguments) of this ReadMe for a full overview of the available input arguments.
A minimal command to train and verify a neural network policy is:

```
python run.py --model <benchmark> --probability_bound <bound> --pretrain_method <algorithm> --pretrain_total_steps <steps>
```

In this command, `<benchmark>` specifies the benchmark to run, `<bound>` is the probability bound of the reach-avoid
specification, and `<algorithm>` is the method used to pretrain the input neural network policy for the specified number
of steps `<steps>` (see below for the options).

For example, to verify a policy trained by PPO (implemented in JAX) for 100K steps on the `linear-sys` benchmark with a
probability bound of $\rho = 0.99$, you run:

```
python run.py --model LinearSystem --probability_bound 0.99 --pretrain_method PPO_JAX --pretrain_total_steps 100000
```

This command stores the trained PPO policy as a checkpoint in the folder `ckpt/` and gives it as input to the
learner-verifier framework.
Upon termination of the framework, the learned certificate is exported to the corresponding subfolder in `logger/`,
together with CSV files that summarize other relevant statistics of the run.

## Validating results

The file `validate_certificate.py` can be used to check the validity of a learned RASM empirically.
This validation can be called automatically upon termination of the learner-verifier by adding the argument `--validate`
to the `run.py` script.
Alternatively, the validation can be called externally on a given checkpoint as follows:

```
python validate_certificate.py --checkpoint 'logger/.../final_ckpt/' --cell_width 0.01
```

Here, `--checkpoint` should be given the path to the exported final checkpoint, and `--cell_width` is the mesh size for
the (uniform) discretization used to validate the RASM empirically.

It is also possible to directly perform the validation for all checkpoints in a given directory:

```
python validate_certificate.py --check_folder 'logger/subfolder-with-multiple-checkpoints/' --cell_width 0.01
```

Several other arguments can be passed; see `validate_certificate.py` for the full overview.

## Training policies with Stable-Baselines

By default, `run.py` trains policies with PPO (implemented in JAX).
For some experiments, we instead train policies with other RL algorithms implemented
in [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/).
Since these implementations are not optimized for our code, we provide a script to externally pretrain policies using
Stable-Baselines3.
This script is called `train_SB3.py` and can, for example, be used as follows:

```
python train_SB3.py --model LinearSystem --layout 0 --algorithm TRPO --total_steps 100000 --seed 1 --num_envs 10 --neurons_per_layer 128 --hidden_layers 3
```

The algorithms we use for our experiments are TRPO, TQC, SAC, and A2C (
see [Section 4](#4.-reproducing-results-from-the-paper) for details).

<br />

# 4. Reproducing results from the paper

You can reproduce the results presented in [1] as follows.
The experiments are divided into several parts, each of which we describe below.
After running the experiments, the provided bash scripts automatically generated the plots and figures as presented in
the paper.

## Ablation study

To reproduce the experiments for the ablation study (i.e., Fig. 2 in [1]), run the following command (excepted run time:
multiple days):

```
bash experiments_linsys.sh > logger/experiments_linsys.out;
bash experiments_pendulum.sh > logger/experiments_pendulum.out;
bash experiments_collisionavoid.sh > logger/experiments_collisionavoid.out;
```

This command runs 3 benchmarks (linear-sys, pendulum, and collision-avoid) for various probability bounds $\rho$, and
with different verifiers.
These verifiers implement different combinations of our techniques, ranging from the *baseline* (all contributions
disabled) to our *new method* (all contributions enabled).
See [1] for more details on each of these verifiers.
After the experiments for a benchmark are finished, the bash script calls `parse_results.py` to collect the results and
generate the figures and tables as in the paper.

## Robustness to input policies

To reproduce the experiments with input policies trained with different RL algorithms (i.e., Fig. 3 in [1]), run the
following command (expected run time: 18 hours):

```
bash experiments_stablebaselines.sh > logger/experiments_stablebaselines.out
```

This command runs 3 benchmarks (linear-sys, pendulum, and collision-avoid) for various probability bounds $\rho$, and
with policies trained using different RL algorithms (namely: TRPO, TQC, SAC, A2C).
We use the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) implementation of these algorithms.
To reduce the run time, we provide these input policies as pretrained checkpoints in this repository (in
the `ckpt_pretrain_sb3/` folder).
After the experiments for a benchmark are finished, the bash script above calls `parse_results.py` to collect the
results and generate the figures and tables as in the paper.

To also recreate the input policies, run the following command:

```
bash train_SB3_all.sh > logger/train_SB3_all.out
```

This script recreates the input policies for all 3 benchmarks, 4 algorithms, and different numbers of steps (1K, 10K,
100K, and 1 million).
The speed differs substantially between the different RL algorithms.
For the slowest algorithms, pretraining a policy for 1 million steps may take a couple of minutes.

## Plotting RASMs

A run of the code is terminated when the verifier succeeded to verify the validity of the learned RASM.
Upon termination, the code automatically plots the RASM, similar to the plots in Fig. 4 in [1].
To reproduce the specific RASMs shown in Fig. 4, run the following commands (combined expected run time: 1 hour):

```
python run.py --model LinearSystem --probability_bound 0.995 --mesh_loss 0.0001 --expDecr_multiplier 10 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --min_lip_policy_loss 0.5 --hidden_layers 3;
python run.py --model LinearSystem --probability_bound 0.99995 --mesh_loss 0.000005 --expDecr_multiplier 10 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --min_lip_policy_loss 0.5 --hidden_layers 3;
python run.py --model LinearSystem --layout 1 --probability_bound 0.9 --mesh_loss 0.0001 --expDecr_multiplier 10 --mesh_verify_grid_init 0.003 --mesh_verify_grid_min 0.003 --min_lip_policy_loss 1 --hidden_layers 3 --layout 1;
```

For each of the three runs, a folder in `logger/` is created, in which the corresponding RASM plot is saved.

## Robustness to network size

The experiments above are run with (certificate and policy) networks with 3 hidden layers of 128 neurons each.
To reproduce the experiments on networks with 2 hidden layers of 128 neurons each (i.e., Fig. 6 in the appendix of [1]),
run the following command (expected run time: 2 days):

```
bash experiments_2layers.sh > logger/experiments_2layers.out
```

After the experiments for a benchmark are finished, this script above calls `parse_results.py` to collect the results
and generate the figures and tables as in the paper.

## Comparison against LipBaB

In the paper, we compare the Lipschitz constants computed using our method with those
from [LipBaB](https://github.com/pyrobits/LipBaB), an anytime algorithm for computing upper bounds on Lipschitz
constants for neural networks.
These experiments can be reproduced by running the command:

```
bash experiments_LipBaB.sh > logger/experiments_LipBaB.out
python3 lipbab_interpret_results.py < logger/experiments_LipBaB.out
```

This script runs LipBaB on several checkpoints of learned RASMs (together with the corresponding policy), which we
provide as pretrained checkpoints in this repository.
The Python script `lipbab_interpret_results.py` then takes the terminal output to produce Table 3 as presented in the
appendix of [1].

## Running the more difficult benchmarks

The experiments on linear-sys with the more difficult reach-avoid specification and on triple-integrator can be
reproduced by running the following command (combined expected run time: 16 hours):

```
bash experiments_scaling.sh > logger/experiments_scaling.out
```

After the experiments for a benchmark are finished, this script above calls `parse_results.py` to collect the results
and generate the figures and tables as in the paper.

<br />

# 6. Overview of input arguments

We provide an overview of the most important input arguments to the `run.py` script.
For an overview of *all arguments*, we refer to `core/parse_args.py` (note that some of these arguments are never
changed from their default values in our experiments).

All arguments are given as `--<argument name> <value>` or (in the case of boolean values) as `--<argument name>`.

## General arguments

| Arguments            | Default   | Help                                                                               |
|----------------------|-----------|------------------------------------------------------------------------------------|
| model                | n/a       | Gymnasium environment ID                                                           |
| layout               | 0         | Select a particular layout for the benchmark model (if this option exists)         |
| probability_bound    | 0.9       | Bound on the reach-avoid probability to verify                                     |
| seed                 | 1         | Random seed                                                                        |
| validate             | FALSE     | If True, automatically perform validation once martingale was successfully learned |
| load_ckpt            | n/a       | If given, a PPO checkpoint in loaded from this file                                |
| pretrain_method      | PPO_JAX   | Method to pretrain (initialize) the policy                                         |
| pretrain_total_steps | 1_000_000 | Total number of timesteps to do with PPO (for policy initialization                |
| logger_prefix        | n/a       | Prefix to logger export file                                                       |
| silent               | FALSE     | Only show crucial output in terminal                                               |
| plot_intermediate    | FALSE     | If True, plots are generated throughout the CEGIS iterations (increases runtime)   |

## Enabling/disabling contributions (as for the ablation)

| Arguments             | Default | Help                                                                                                         |
|-----------------------|---------|--------------------------------------------------------------------------------------------------------------|
| local_refinement      | TRUE    | If True, local grid refinements are performed                                                                |
| weighted              | TRUE    | If True, use weighted norms to compute Lipschitz constants                                                   |
| cplip                 | TRUE    | If True, use CPLip method to compute Lipschitz constants                                                     |
| improved_softplus_lip | TRUE    | If True, use improved (local) Lipschitz constants for softplus in V (if False, global constant of 1 is used) |

## Learner arguments

| Learner                       | Default  | Help                                                                      |
|-------------------------------|----------|---------------------------------------------------------------------------|
| Policy_learning_rate          | 5,00E-05 | Learning rate for changing the policy in the CEGIS loop                   |
| V_learning_rate               | 5,00E-04 | Learning rate for changing the certificate in the CEGIS loop              |
| cegis_iterations              | 1000     | Number of CEGIS iteration to run                                          |
| epochs                        | 25       | Number of epochs to run in each iteration                                 |
| num_samples_per_epoch         | 90000    | Total number of samples to train over in each epoch                       |
| num_counterexamples_in_buffer | 30000    | Total number of samples to train over in each epoch                       |
| batch_size                    | 4096     | Batch size used by the learner in each epoch                              |
| expDecr_multiplier            | 1        | Multiply the weighted counterexample expected decrease loss by this value |

## Verifier arguments

| Verifier                  | Default   | Help                                                                                                                    |
|---------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------|
| mesh_loss                 | 0.001     | Mesh size used in the loss function                                                                                     |
| mesh_verify_grid_init     | 0.01      | Initial mesh size for verifying grid. Mesh is defined such that \|x-y\|_1 <= tau for any x in X and discretized point y |
| mesh_verify_grid_min      | 0.01      | Minimum mesh size for verifying grid                                                                                    |
| mesh_refine_min           | 1,00E-09  | Lowest allowed verification grid mesh size in the final verification                                                    |
| max_refine_factor         | 10        | Maximum value to split each grid point into (per dimension), during the (local) refinement                              |
| verify_batch_size         | 30000     | Number of states for which the verifier checks exp. decrease condition in the same batch.                               |
| forward_pass_batch_size   | 1_000_000 | Batch size for performing forward passes on the neural network (reduce if this gives memory issues).                    |
| noise_partition_cells     | 12        | Number of cells to partition the noise space in per dimension (to numerically integrate stochastic noise)               |
| counterx_refresh_fraction | 0.50      | Fraction of the counter example buffer to renew after each iteration                                                    |
| counterx_fraction         | 0.25      | Fraction of counter examples, compared to the total train data set.                                                     |
