
import matplotlib.pyplot as plt

label = {'acc' : 'Accuracy',
         'parity' : 'Dem. parity',
         'mono_yexp' : 'Mono(yexp)',
         'mono_hpw' : 'Mono(hpw)',
         'mono_hw' : 'Mono(hw)',
         'rob_yexp' : 'Robust(yexp)'}

color = {'acc' : 'C0',
         'parity' : 'C1',
         'rob_yexp' : 'C2',
         'mono_yexp' : 'C3',
         'mono_hw' : 'C4',
         'mono_hpw' : 'C6'}

linestyle = {'acc' : 'solid',
             'parity' : 'solid',
             'rob_yexp' : 'solid',
             'mono_yexp' : 'dotted',
             'mono_hw' : 'dashed',
             'mono_hpw' : 'dashdot'}

def plot(data, plot_path):

    modes = ['unbiased', 'biased']
    curves = ['acc', 'parity', 'rob_yexp', 'mono_yexp', 'mono_hpw', 'mono_hw']

    legend_fontsize = 16
    fontsize = 20
    alpha = 0.9
    linewidth = 4
    ylimits = (0.4, 1.2)
    figsize = (10,10)

    plt.rcParams.update({'font.size': fontsize})
    fig = plt.figure(figsize=figsize)
    axis = {}
    for i, mode in enumerate(modes):
        ax = fig.add_subplot(1, 2, i+1)
        #ax.set_aspect(aspect[0]/aspect[1])
        ax.set_xlabel(f'Epoch({mode})')
        ax.set_ylim(*ylimits)

        if i > 0:
            ax.set_yticklabels([])

        axis[mode] = ax

    for mode in modes:
        x = [r['epoch'] for r in data[mode]]
        for curve in curves:
            y = [r[curve] for r in data[mode]]
            maybelabel = label[curve] if mode == 'unbiased' else None
            axis[mode].plot(x, y,
                            color=color[curve],
                            label=maybelabel,
                            linestyle=linestyle[curve],
                            linewidth=linewidth,
                            alpha=alpha)

    fig.legend(fontsize=legend_fontsize, ncol=len(curves)/2, loc='upper center')
    #plt.legend(loc='upper center', #bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=len(curves)/2)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':

    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('unbiased_path', type=str, help="Unbiased results path")
    parser.add_argument('biased_path', type=str, help="Biased results path")
    parser.add_argument('--plot_path', type=str, help="Plot path", default='income_plot.png')
    args = parser.parse_args()

    data = {}
    try:
        with open(args.unbiased_path, 'rb') as f:
            data['unbiased'] = pickle.load(f)
        with open(args.biased_path, 'rb') as f:
            data['biased'] = pickle.load(f)
    except FileNotFoundError:
        print("Couldn't open the results file")
        exit(1)

    plot(data, args.plot_path)

