from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

font = {'size': 11}
matplotlib.rc('font', **font)

colors = {
    1: [55 / 255, 126 / 255, 184 / 255],  # Lightblueish
    2: [228 / 255, 26 / 255, 28 / 255],  # Red
    3: [77 / 255, 175 / 255, 74 / 255],  # green
    4: [152 / 255, 78 / 255, 163 / 255],  # purple
    5: [255 / 255, 127 / 255, 0 / 255],  # orange
    6: [0.5, 1.0, 0.83],  # aquamarine
    7: [1.0, 0.0, 1.0],  # magenta
    8: [0.66, 0.66, 0.66],  # gray
}


def bar_plot(plot_x, cases, df, cwd, export_file, timeout=3600):
    '''
    Generate bar plots as presented in the paper
    
    :param plot_x: xticks for plot
    :param cases: list of categories to plot
    :param df: dataframe
    :param cwd: current working directory (str)
    :param export_file: file to export plot to
    :param timeout: timeout (3600 seconds by default)
    :return:
    '''

    length = len(plot_x)

    spacing = 0.9
    x = np.arange(length) * spacing
    width = 0.1
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))

    ymax = timeout + 50

    for i, case in enumerate(cases):
        if case + '_AM' not in df:
            continue

        # Set nan's to 3600 sec (value of timeout)
        values = np.array(df[case + '_AM'])
        values[np.isnan(values)] = timeout

        error = df[case + '_STD']

        # Set variable width (smaller bars for NaN results)
        W = [0.8 * width if v >= timeout else 0.8 * width for v in values]
        C = [colors[i + 1] + [1] if v >= timeout else colors[i + 1] + [1] for v in values]

        offset = width * multiplier

        # Add normal bars
        rects = ax.bar(x - offset, values, width=W, label=case, color=C)
        errbar = ax.errorbar(x - offset, values, yerr=error, color='dimgray', fmt='x', markersize=0.01, ls='none',
                             capsize=3)

        multiplier += 1

    ax.set_ylabel('Time [s]')
    ax.set_xticks(x - length * 0.5 * width, plot_x)
    ax.legend(loc='upper left', ncols=3)

    # Set axis range
    ax.set_ylim(0, ymax)
    yvalues = np.arange(0, timeout + 1, 600)
    ax.set_yticks(yvalues)
    ylabels = list(yvalues[:-1].astype(str)) + ['TO/MO']
    ax.set_yticklabels(ylabels)

    def tikzplotlib_fix_ncols(obj):
        """
        workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
        """
        if hasattr(obj, "_ncols"):
            obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)

    plt.legend(ncol=1, loc='upper left')

    tikzplotlib_fix_ncols(fig)

    # plt.savefig('figure.pdf', backend='pgf')
    tikzplotlib.save(Path(cwd, export_file + '_barplot.tex'), strict=True)

    # Save figure
    for form in ['pdf', 'png']:
        filepath = Path(cwd, export_file).with_suffix('.' + str(form))
        plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)

    plt.show()
