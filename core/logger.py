import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.jax_utils import lipschitz_coeff


class Logger:
    ''' Class to export results to csv files '''

    def __init__(self, logger_folder, round_decimals=-1):

        # Create logger folder
        Path(logger_folder).mkdir(parents=True, exist_ok=True)

        # Set an initial timer
        self.set_timer()

        self.round_decimals = round_decimals

        self.times = {}
        self.times_series = pd.Series(self.times, dtype=np.float32)
        self.Lipschitz_df = pd.DataFrame({}, dtype=np.float32)

        self.logger_folder = logger_folder

        self.info_path = Path(self.logger_folder, 'info.csv')
        self.times_path = Path(self.logger_folder, 'times.csv')
        self.args_path = Path(self.logger_folder, 'args.csv')
        self.Lipschitz_path = Path(self.logger_folder, 'lipschitz.csv')

        self.violations_per_iteration = {}
        self.info = {}

        return

    def export_args(self, args):

        dict = vars(args)
        args_series = pd.Series(dict, name="arguments")
        args_series.to_csv(self.args_path)

    def set_timer(self):
        '''
        Store current time
        '''

        self.curr_time = time.time()

    def get_timer_value(self, reset=True):
        '''
        Compute time elapsed since time was last stored. If reset=True, then the timer is reset afterward.
        '''

        time_elapsed = time.time() - self.curr_time

        if reset:
            self.set_timer()

        return time_elapsed

    def append_time(self, key, value, export=True):

        if self.round_decimals >= 0:
            value = np.around(value, decimals=self.round_decimals)

        if key in self.times:
            print(f'(!) Logger warning: key "{key}" already exists in timing dictionary')
        self.times[key] = value

        if export:
            self.export_times()

        return

    def append_Lipschitz(self, Policy_state, V_state, iteration, silent, export=True):

        t = time.time()

        lips_pi = {}
        lips_V = {}

        # Compute all Lipschitz constants
        for weighted in [True, False]:
            for CPLip in [True, False]:
                key1 = f'pi-lip_w={str(weighted)}_CPLip={str(CPLip)}'
                key2 = f'pi-weights_w={str(weighted)}_CPLip={str(CPLip)}'
                lips_pi[key1], lips_pi[key2] = lipschitz_coeff(Policy_state.params, weighted, CPLip, False)

        # Convert to DataFrame
        pi_DF = pd.DataFrame([lips_pi], index=[int(iteration)])
        pi_DF.index.name = 'iter'

        if not silent:
            print('\nPolicy Lipschitz constants:')
            print(pi_DF)

        for weighted in [True, False]:
            for CPLip in [True, False]:
                key1 = f'V-lip_w={str(weighted)}_CPLip={str(CPLip)}'
                key2 = f'V-weights_w={str(weighted)}_CPLip={str(CPLip)}'
                lips_V[key1], lips_V[key2] = lipschitz_coeff(V_state.params, weighted, CPLip, False)

        # Convert to DataFrame
        V_DF = pd.DataFrame([lips_V], index=[int(iteration)])
        V_DF.index.name = 'iter'

        if not silent:
            print('\nCertificate Lipschitz constants:')
            print(V_DF)

        # Append to main DataFrame
        add_df = pd.concat([pi_DF, V_DF], axis=1)
        self.Lipschitz_df = pd.concat([self.Lipschitz_df, add_df])

        if export:
            self.export_Lipschitz()

        print(f'Time to compute and export Lipschitz constants: {time.time() - t:.4f} sec.')

        return

    def export_times(self):

        self.times_series = pd.Series(self.times, dtype=np.float32, name="time [s]")
        self.times_series.to_csv(self.times_path)

        return

    def export_Lipschitz(self):

        self.Lipschitz_df.to_csv(self.Lipschitz_path)

        return

    def add_info_from_dict(self, dict, export=True):
        ''' Add info from a dictionary '''

        for key, value in dict.items():
            self.add_info(key, value, export=False)

        if export:
            self.export_info()

        return

    def add_info(self, key, value, export=True):
        ''' Add single element to info dictionary '''

        if type(value) != str and type(value) != bool and self.round_decimals >= 0:
            value = np.around(value, decimals=self.round_decimals)

        if key in self.info:
            print(f'(!) Logger warning: key "{key}" already exists in info dictionary')
        self.info[key] = value

        if export:
            self.export_info()

        return

    def export_info(self):

        self.times_series = pd.Series(self.info, name="info")
        self.times_series.to_csv(self.info_path)

        return
