#
# SER performance for different Delta (measurement ratio) in binary vector reconstruction (Fig. 6)
#

import numpy as np
from tqdm import tqdm
from my_module import cs, cgmt, my_plot


def main():
    # problem settings
    array_delta = np.array([0.7, 0.8, 0.9])  # measurement ratio
    prob_param = cs.ProbParam(N=500,  # dimension of the unknown vector
                              distribution='binary',  # distribution of the unknown vector
                              sigma2_v=4e-2)  # noise variance

    # algorithm settings
    alg_param = cs.AlgParam(rho=0.1,  # parameter of ADMM
                            prox=cs.prox_box,  # proximity operator
                            num_iteration=20)  # number of iterations

    num_empirical = 300  # number of samples for the empirical performance evaluation (>= 100 is better)
    sample_size = 300000  # sample size for the prediction (>= 100000 is better)
    measure = 'SER'  # performance measure

    dict_array_SER_prediction = {}
    dict_array_SER_empirical = {}
    for delta in tqdm(array_delta, desc='simulation progress', colour='blue', ncols=80):
        prob_param.delta = delta
        # performance prediction
        dict_array_SER_prediction[delta] = cgmt.state_evolution(prob_param, alg_param, sample_size=sample_size, measure=measure)
        # empirical performance
        dict_array_SER_empirical[delta] = cs.empirical_performance(prob_param, alg_param, num_empirical=num_empirical, measure=measure)

    # plot results
    fig, ax, array_marker, array_color = my_plot.setup()
    line_width = 2
    marker_size = 10

    for i, delta in enumerate(array_delta):
        ax.plot(range(alg_param.num_iteration + 1), dict_array_SER_empirical[delta],
                label=rf'empirical ($\Delta={delta}$)', linestyle='', color=array_color[i], marker=array_marker[i], markersize=marker_size)
    for i, delta in enumerate(array_delta):
        if i == 0:
            ax.plot(range(alg_param.num_iteration + 1), dict_array_SER_prediction[delta],
                    label='prediction', linestyle='-', color=array_color[i], linewidth=line_width)
        else:
            ax.plot(range(alg_param.num_iteration + 1), dict_array_SER_prediction[delta],
                    linestyle='-', color=array_color[i], linewidth=line_width)

    my_plot.set_ax_property(ax,
                            yscale='log',
                            xticks=range(0, alg_param.num_iteration + 1, 5),
                            xlim_left=0,
                            xlim_right=alg_param.num_iteration,
                            ylim_top=1,
                            ylim_bottom=1e-4,
                            xlabel='number of iterations',
                            ylabel=measure)
    fig.savefig('figure/SER_vs_iteration_Delta(binary).pdf')


if __name__ == '__main__':
    main()
