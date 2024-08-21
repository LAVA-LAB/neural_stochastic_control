import argparse

import numpy as np


def parse_arguments(linfty, datetime, cwd):
    """
    Function to parse arguments provided

    Returns
    -------
    :args: Dictionary with all arguments

    """

    # GENERAL OPTIONS
    parser = argparse.ArgumentParser(prefix_chars='--')
    parser.add_argument('--model', type=str, default="LinearSystem",
                        help="Gymnasium environment ID")
    parser.add_argument('--layout', type=int, default=0,
                        help="Select a particular layout for the benchmark model (if this option exists)")
    parser.add_argument('--probability_bound', type=float, default=0.9,
                        help="Bound on the reach-avoid probability to verify")
    parser.add_argument('--seed', type=int, default=1,
                        help="Random seed")
    parser.add_argument('--logger_prefix', type=str, default="",
                        help="Prefix to logger export file")
    parser.add_argument('--silent', action=argparse.BooleanOptionalAction, default=False,
                        help="Only show crucial output in terminal")
    parser.add_argument('--update_certificate', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, certificate network is updated by the Learner")
    parser.add_argument('--update_policy', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, policy network is updated by the Learner")
    parser.add_argument('--plot_intermediate', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plots are generated throughout the CEGIS iterations (increases runtime)")
    parser.add_argument('--presentation_plots', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, labels and titles are omitted from plots (better for generating GIFs for, e.g., presentations)")
    parser.add_argument('--validate', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, automatically perform validation once martingale was successfully learned")

    ### POLICY INITIALIZATION ARGUMENTS
    parser.add_argument('--load_ckpt', type=str, default='',
                        help="If given, a PPO checkpoint in loaded from this file")
    parser.add_argument('--pretrain_method', type=str, default='PPO_JAX',
                        help="Method to pretrain (initialize) the policy")
    parser.add_argument('--pretrain_total_steps', type=int, default=1_000_000,
                        help="Total number of timesteps to do with PPO (for policy initialization")
    parser.add_argument('--pretrain_num_envs', type=int, default=10,
                        help="Number of parallel environments in PPO (for policy initialization")

    ### JAX PPO arguments
    parser.add_argument('--ppo_max_policy_lipschitz', type=float, default=3,
                        help="Max. Lipschitz constant for policy to train towards in PPO (below this value, loss is zero)")
    parser.add_argument('--ppo_num_steps_per_batch', type=int, default=2048,
                        help="Total steps for rollout in PPO (for policy initialization")
    parser.add_argument('--ppo_num_minibatches', type=int, default=32,
                        help="Number of minibatches in PPO (for policy initialization")
    parser.add_argument('--ppo_verbose', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, print more output during PPO (JAX) training")

    ### VERIFY MESH SIZES
    parser.add_argument('--mesh_loss', type=float, default=0.001,
                        help="Mesh size used in the loss function")
    parser.add_argument('--mesh_verify_grid_init', type=float, default=0.01,
                        help="Initial mesh size for verifying grid. Mesh is defined such that |x-y|_1 <= tau for any x in X and discretized point y")
    parser.add_argument('--mesh_verify_grid_min', type=float, default=0.01,
                        help="Minimum mesh size for verifying grid")

    ### REFINE ARGUMENTS
    parser.add_argument('--mesh_refine_min', type=float, default=1e-9,
                        help="Lowest allowed verification grid mesh size in the final verification")
    parser.add_argument('--max_refine_factor', type=float, default=float('nan'),
                        help="Maximum value to split each grid point into (per dimension), during the (local) refinement")

    ### LEARNING RATES
    parser.add_argument('--Policy_learning_rate', type=float, default=5e-5,
                        help="Learning rate for changing the policy in the CEGIS loop")
    parser.add_argument('--V_learning_rate', type=float, default=5e-4,
                        help="Learning rate for changing the certificate in the CEGIS loop")

    ### LEARNER ARGUMENTS
    parser.add_argument('--cegis_iterations', type=int, default=1000,
                        help="Number of CEGIS iteration to run")
    parser.add_argument('--epochs', type=int, default=25,
                        help="Number of epochs to run in each iteration")
    parser.add_argument('--num_samples_per_epoch', type=int, default=90000,
                        help="Total number of samples to train over in each epoch")
    parser.add_argument('--num_counterexamples_in_buffer', type=int, default=30000,
                        help="Total number of samples to train over in each epoch")
    parser.add_argument('--batch_size', type=int, default=4096,
                        help="Batch size used by the learner in each epoch")
    parser.add_argument('--loss_lipschitz_lambda', type=float, default=0,
                        help="Factor to multiply the Lipschitz loss component with")
    parser.add_argument('--loss_lipschitz_certificate', type=float, default=15,
                        help="When the certificate Lipschitz coefficient is below this value, then the loss is zero")
    parser.add_argument('--loss_lipschitz_policy', type=float, default=4,
                        help="When the policy Lipschitz coefficient is below this value, then the loss is zero")
    parser.add_argument('--expDecr_multiplier', type=float, default=1,
                        help="Multiply the weighted counterexample expected decrease loss by this value.")
    parser.add_argument('--debug_train_step', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, generate additional plots for the samples used in the last train step of an iteration")
    parser.add_argument('--auxiliary_loss', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, auxiliary loss is added to the learner loss function")

    ### VERIFIER ARGUMENTS
    parser.add_argument('--verify_batch_size', type=int, default=30000,
                        help="Number of states for which the verifier checks exp. decrease condition in the same batch.")
    parser.add_argument('--forward_pass_batch_size', type=int, default=1_000_000,
                        help="Batch size for performing forward passes on the neural network (reduce if this gives memory issues).")
    parser.add_argument('--noise_partition_cells', type=int, default=12,
                        help="Number of cells to partition the noise space in per dimension (to numerically integrate stochastic noise)")
    parser.add_argument('--counterx_refresh_fraction', type=float, default=0.50,
                        help="Fraction of the counter example buffer to renew after each iteration")
    parser.add_argument('--counterx_fraction', type=float, default=0.25,
                        help="Fraction of counter examples, compared to the total train data set.")

    ### ARGUMENTS TO EXPERIMENT WITH ###
    parser.add_argument('--local_refinement', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, local grid refinements are performed")
    parser.add_argument('--weighted_counterexamples', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, weighted counterexamples are used in the learner loss function")
    parser.add_argument('--perturb_counterexamples', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, counterexamples are perturbed before being added to the counterexample buffer")
    parser.add_argument('--min_lip_policy_loss', type=float, default=0,
                        help="Minimum Lipschitz constant policy used in loss function learner.")
    parser.add_argument('--hard_violation_multiplier', type=float, default=10,
                        help="Factor to multiply the counterexample weights for hard violations with.")
    parser.add_argument('--weighted_counterexample_sampling', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, use weighted sampling of counterexamples")
    parser.add_argument('--min_fraction_samples_per_region', type=float, default=0,
                        help="Minimum fraction of samples in learner for each region/condition.")

    ### NEURAL NETWORK ARCHITECTURE
    parser.add_argument('--neurons_per_layer', type=int, default=128,
                        help="Number of neurons per (hidden) layer.")
    parser.add_argument('--hidden_layers', type=int, default=2,
                        help="Number of hidden layers.")

    ## LIPSCHITZ COEFFICIENT ARGUMENTS
    parser.add_argument('--weighted', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, use weighted norms to compute Lipschitz constants")
    parser.add_argument('--cplip', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, use CPLip method to compute Lipschitz constants")
    parser.add_argument('--split_lip', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, use L_f split over the system state space and control action space")
    parser.add_argument('--improved_softplus_lip', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, use improved (local) Lipschitz constants for softplus in V (if False, "
                             "global constant of 1 is used)")

    args = parser.parse_args()

    args.linfty = linfty  # Use L1 norm for Lipschitz constants (there is unused but experimental support for L_infty norms)
    args.start_datetime = datetime
    args.cwd = cwd

    # Set refinement factor depending on whether local refinement is enabled
    if args.local_refinement and args.max_refine_factor != args.max_refine_factor:  # a != a means that a is NaN
        args.max_refine_factor = 10
    elif not args.local_refinement and args.max_refine_factor != args.max_refine_factor:  # a != a means that a is NaN
        args.max_refine_factor = np.sqrt(2)

    return args
