import argparse
import os
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

from core.bar_plot import bar_plot


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, help="Folder to parse.")
parser.add_argument('--sb', action='store_true', default=False, required=False,
                    help="If true, generate results for Stable-Baselines3 experiments.")
parser.add_argument('--plot_x', type=str, action="store", dest='plot_x',
                    default='[]', help="Values of x (if not provided, no plot is created)")
parser.add_argument('--include_num_samples', action='store_true', default=False, required=False,
                    help="If True, the number of samples (for our local scheme vs. baseline) is included in the table.")
parser = parser.parse_args()

parser.plot_x = list(literal_eval(parser.plot_x))  # Interpret as list
parser.plot_x = [str(x) for x in parser.plot_x]  # Convert to strings

print(f'- Run for folder: {parser.folder}')
if len(parser.plot_x) > 0:
    print(f'- Create bar plot with x-values: {parser.plot_x}')
else:
    print('-Do not create bar plot')

# if benchmark == 'linsys':
#     parser.folder = '../neurips_results/linsys_ablation'
#     plot_x = ['0.995', '0.9995', '0.99995']  # For linsys
# elif benchmark == 'linsys_layout1':
#     parser.folder = '../neurips_results/linsys_layout1'
#     plot_x = [git pu]
#     parser.include_num_samples = True
# elif benchmark == 'triple_integrator':
#     parser.folder = '../neurips_results/tripleintegrator'
#     plot_x = ['0.8', '0.9', '0.95']
#     parser.include_num_samples = True
# elif benchmark == 'pendulum':
#     parser.folder = '../neurips_results/pendulum_ablation'
#     plot_x = ['0.9', '0.95', '0.99']  # For pendulum
# elif benchmark == 'collision':
#     parser.folder = '../neurips_results/collision_ablation'
#     plot_x = ['0.9', '0.95', '0.98']  # For collision avoid
# elif benchmark == 'linsys_sb':
#     parser.folder = '../neurips_results/sb/linsys_sb'
#     parser.sb = True
#     plot_x = ['10000', '100000', '1000000']
# elif benchmark == 'pendulum_sb':
#     parser.folder = '../neurips_results/sb/pendulum_sb'
#     parser.sb = True
#     plot_x = ['10000', '100000', '1000000']
# elif benchmark == 'collision_sb':
#     parser.folder = '../neurips_results/sb/collision_sb'
#     parser.sb = True
#     plot_x = ['10000', '100000', '1000000']
# elif benchmark == 'nn_ablation_linsys':
#     parser.folder = '../neurips_results/nn_ablation/linsys'
#     plot_x = ['0.995', '0.9995', '0.99995']
# elif benchmark == 'nn_ablation_pendulum':
#     parser.folder = '../neurips_results/nn_ablation/pendulum'
#     plot_x = ['0.9', '0.95', '0.99']
# elif benchmark == 'nn_ablation_collision':
#     parser.folder = '../neurips_results/nn_ablation/collision'
#     plot_x = ['0.9', '0.95', '0.98']
# else:
#     assert False;
#     'Invalid case given'

#####

# Parameters
input_folder = parser.folder
export_file = parser.folder
decimals = 0

cwd = os.getcwd()

PATH = Path(cwd, input_folder)

subfolders = [f.path for f in os.scandir(PATH) if f.is_dir()]
validate_ckpts = []

if parser.sb:
    num_seeds = 5
    seed_list = [str(i) for i in range(num_seeds)]

    stat_cols = ['fails', 'AM', 'STD']

    experiment_params = ['algo', 'steps']
    DF = pd.DataFrame(columns=['model'] + experiment_params + seed_list + stat_cols)

    DF.index.name = 'i'
    DIC = {}

    print(f'- Found {len(subfolders)} experiment instances to parse')

    # Iterate over all folders
    for folder in subfolders:
        info_file = Path(folder, 'info.csv')

        info = pd.read_csv(info_file, index_col=0)['info']

        args_file = Path(folder, 'args.csv')
        argsf = pd.read_csv(args_file, index_col=0)['arguments']
        ckpt = argsf['load_ckpt']
        idx_alg = ckpt.find("alg=") + 4
        idx_algend = ckpt[idx_alg:].find("_") + idx_alg
        alg = ckpt[idx_alg:idx_algend]
        steps = ckpt[-ckpt[::-1].find("="):]
        key = (info['model'], alg, steps)

        seed = int(info['seed'])

        if 'total_CEGIS_time' in info:
            time = float(info['total_CEGIS_time'])
        else:
            time = np.nan

        if not key in DIC:
            DIC[key] = {'times': np.full(num_seeds, fill_value=-1, dtype=float), 'infos': {}}

        if DIC[key]['times'][seed - 1] != -1:
            print(f'- Warning, overwriting value for key: {key} values: {DIC[key][seed - 1]} --> {time}')

        DIC[key]['times'][seed - 1] = float(time)
        DIC[key]['infos'][seed - 1] = info

    print(f'- Parsed into dictionary')

    plot_dic = {}

    for (model, algo, steps), item in DIC.items():

        times = item['times']
        infos = item['infos']

        # Compute number of fails
        fails = int(np.sum(np.isnan(times) + (times < 0)))

        # Set all nan times to 3600
        times[np.isnan(times) + (times < 0)] = 1800

        if fails > 1:
            # GM = 1800
            AM = 1800
            STD = np.nan
        else:
            # GM = np.round(geo_mean(times), decimals=decimals) if len(times) > 0 else np.nan
            AM = np.round(np.mean(times), decimals=decimals) if len(times) > 0 else np.nan
            STD = np.round(np.std(times), decimals=decimals) if len(times) > 0 else np.nan

        row = [model, algo, steps] + list(np.round(times, decimals=0)) + [fails, AM, STD]

        DF.loc[len(DF), DF.columns] = row

        # Create column in table used for plotting graphs
        if algo + '_AM' not in plot_dic:
            # plot_dic[algo + '_GM'] = {}
            plot_dic[algo + '_AM'] = {}
            plot_dic[algo + '_STD'] = {}

        # p_log = np.round(-np.log(1-float(p))/np.log(10), 2)
        # plot_dic[algo + '_GM'][steps] = GM
        plot_dic[algo + '_AM'][steps] = AM
        plot_dic[algo + '_STD'][steps] = STD

    # Merge into single DataFrame
    plot_df = pd.DataFrame(plot_dic)
    plot_df.index.name = 'steps'  # Set index name
    plot_df.sort_index(axis=0, ascending=True, inplace=True)  # Sort by steps
    plot_df['x'] = np.arange(len(plot_df)) + 1  # Add categorical labels for bar plots
    plot_df.to_csv(Path(cwd, export_file + '_plot.csv'))

    # Create bar plot
    if len(parser.plot_x) > 0:
        ALL_CASES = ['A2C', 'SAC', 'TQC', 'TRPO']
        bar_plot(parser.plot_x, ALL_CASES, plot_df.loc[parser.plot_x], cwd, export_file, timeout=1800)

    # Sort values
    DF.sort_values(['algo', 'steps'], ascending=[False, True], inplace=True)

    print(f'- Parsed into DataFrame')

    export_path = Path(cwd, export_file + '.csv')
    DF.to_csv(export_path)
    print(f'- Exported to CSV: {export_file}.csv')


else:
    num_seeds = 5
    seed_list = [str(i) for i in range(num_seeds)]

    stat_cols = ['fails', 'AM', 'STD']

    experiment_params = ['weighted', 'cplip', 'softplus_lip', 'local_refinement']

    columns = ['model', 'case', 'probability'] + stat_cols
    if parser.include_num_samples:
        columns += ['local_AM', 'naive_AM']
    DF = pd.DataFrame(columns=columns)

    DF.index.name = 'i'
    DIC = {}

    DF_plot = pd.DataFrame()

    print(f'- Found {len(subfolders)} experiment instances to parse')

    # Iterate over all folders
    for folder in subfolders:
        info_file = Path(folder, 'info.csv')

        info = pd.read_csv(info_file, index_col=0)['info']

        key = (info['model'], info['weighted_Lipschitz'], info['cplip'], info['improved_softplus_lip'],
               info['local_refinement'], info['probability_bound'])

        seed = int(info['seed'])

        if 'total_CEGIS_time' in info:
            time = float(info['total_CEGIS_time'])
        else:
            time = np.nan

        if not key in DIC:
            DIC[key] = {'times': np.full(num_seeds, fill_value=-1, dtype=float), 'infos': {}}

        if DIC[key]['times'][seed - 1] != -1:
            print(f'- Warning, overwriting value for key: {key} values: {DIC[key][seed - 1]} --> {time}')

        DIC[key]['times'][seed - 1] = float(time)
        DIC[key]['infos'][seed - 1] = info

    print(f'- Parsed into dictionary')

    plot_dic = {}

    for (model, weighted, cplip, softplus_lip, local_refinement, p), item in DIC.items():

        times = item['times']
        infos = item['infos']

        if weighted == 'True' and cplip == 'True' and softplus_lip == 'True' and local_refinement == 'True':
            case = 'our method'
        elif weighted == 'False' and cplip == 'True' and softplus_lip == 'True' and local_refinement == 'True':
            case = 'w/o weighted'
        elif weighted == 'True' and cplip == 'False' and softplus_lip == 'True' and local_refinement == 'True':
            case = 'w/o cplip'
        elif weighted == 'True' and cplip == 'True' and softplus_lip == 'False' and local_refinement == 'True':
            case = 'w/o softplus-lip'
        elif weighted == 'False' and cplip == 'False' and softplus_lip == 'False' and local_refinement == 'True':
            case = 'w/o Lipschitz'
        elif weighted == 'True' and cplip == 'True' and softplus_lip == 'True' and local_refinement == 'False':
            case = 'w/o local'
        elif weighted == 'False' and cplip == 'False' and softplus_lip == 'False' and local_refinement == 'False':
            case = 'baseline'
        else:
            case = 'UNKNOWN'

        # Compute number of fails
        fails = int(np.sum(np.isnan(times) + (times < 0)))

        # Set all nan times to 3600
        times[np.isnan(times) + (times < 0)] = 3600

        if fails > 1:
            # GM = 3600
            AM = 3600
            STD = np.nan
        else:
            # GM = np.round(geo_mean(times), decimals=decimals) if len(times) > 0 else np.nan
            AM = int(np.round(np.mean(times), decimals=decimals)) if len(times) > 0 else np.nan
            STD = int(np.round(np.std(times), decimals=decimals)) if len(times) > 0 else np.nan

        row = [model, case, p] + [fails, AM, STD]

        if parser.include_num_samples:
            # Include naive vs. local sample count (average and stdev) in the table
            local = [int(info['verify_samples']) for info in infos.values() if 'verify_samples' in info]

            # local_gm = int(geo_mean(local))
            local_am = int(np.mean(local))
            local_std = int(np.std(local))

            # Include naive vs. local sample count (average and stdev) in the table
            naive = [int(info['verify_samples_naive']) for info in infos.values() if 'verify_samples_naive' in info]
            # naive_gm = int(geo_mean(naive))
            naive_am = int(np.mean(naive))
            naive_std = int(np.std(naive))

            row = row + [f'{local_am:.2E}', f'{naive_am:.2E}']

        DF.loc[len(DF), DF.columns] = row

        # Create column in table used for plotting graphs
        if case + '_AM' not in plot_dic:
            # plot_dic[case + '_GM'] = {}
            plot_dic[case + '_AM'] = {}
            plot_dic[case + '_STD'] = {}

        # p_log = np.round(-np.log(1-float(p))/np.log(10), 2)
        # plot_dic[case + '_GM'][p] = GM
        plot_dic[case + '_AM'][p] = AM
        plot_dic[case + '_STD'][p] = STD

    # Merge into single DataFrame
    plot_df = pd.DataFrame(plot_dic)
    export_path = Path(cwd, export_file + '_plot.csv')
    plot_df.index.name = 'p'  # Set index name
    plot_df.sort_index(axis=0, ascending=True, inplace=True)  # Sort by probability bound
    plot_df['x'] = np.arange(len(plot_df)) + 1  # Add categorical labels for bar plots
    plot_df.to_csv(export_path)

    # Create bar plot
    if len(parser.plot_x) > 0:
        ALL_CASES = ['our method', 'w/o cplip', 'w/o softplus-lip', 'w/o weighted', 'w/o Lipschitz', 'w/o local',
                     'baseline']
        bar_plot(parser.plot_x, ALL_CASES, plot_df.loc[parser.plot_x], cwd, export_file)

    # Sort values
    DF.sort_values(['case', 'probability'], ascending=[False, True], inplace=True)

    export_path = Path(cwd, export_file + '.tex')
    DF.drop(columns=['case', 'fails']).to_latex(export_path, index=False)
    print(f'- Exported to LaTeX table: {export_file}.tex')

    print(f'- Parsed into DataFrame')

    export_path = Path(cwd, export_file + '.csv')
    DF.to_csv(export_path)
    print(f'- Exported to CSV: {export_file}.csv')
