#
# MSE performance for different N (dimension of the unknown vector) in sparse vector reconstruction (Fig. 2)
#

import numpy as np
from tqdm import tqdm
from my_module import cs, cgmt, my_plot


def main():
    # problem settings
    array_N = np.array([50, 100, 500, 1000])  # dimension of the unknown vector
    prob_param = cs.ProbParam(delta=0.9,  # measurement ratio
                              distribution='sparse',  # distribution of the unknown vector
                              p0=0.8,  # sparsity of the unknown vector
                              sigma2_v=1e-3)  # noise variance

    # algorithm settings
    alg_param = cs.AlgParam(rho=0.1,  # parameter of ADMM
                            prox=cs.prox_L1,  # proximity operator
                            num_iteration=20)  # number of iterations

    num_empirical = 100  # number of samples for the empirical performance evaluation (>= 100 is better)
    sample_size = 100000  # sample size for the prediction (>= 100000 is better)

    # performance prediction
    MSE_opt, alg_param.lmd = cgmt.optimal_performance(prob_param, sample_size=sample_size)
    array_MSE_prediction = cgmt.state_evolution(prob_param, alg_param, sample_size=sample_size, leave=True)

    # empirical performance
    dict_array_MSE_empirical = {}
    for N in tqdm(array_N, desc='empirical performance', colour='blue', ncols=80):
        prob_param.N = N
        dict_array_MSE_empirical[N] = cs.empirical_performance(prob_param, alg_param, num_empirical=num_empirical)

    # plot results
    fig, ax, array_marker, _ = my_plot.setup()
    line_width = 2
    marker_size = 10

    for i, N in enumerate(array_N):
        ax.plot(range(alg_param.num_iteration + 1), dict_array_MSE_empirical[N],
                label=rf'empirical (${N=}$)', linestyle='', marker=array_marker[i], markersize=marker_size)
    ax.plot(range(alg_param.num_iteration + 1), array_MSE_prediction,
            label='prediction', linestyle='-', color='k', linewidth=line_width)
    ax.hlines(y=MSE_opt, xmin=0, xmax=alg_param.num_iteration,
              label='asymptotic MSE of optimizer', linestyle='--', color='k', linewidth=line_width)

    my_plot.set_ax_property(ax,
                            yscale='log',
                            xticks=range(0, alg_param.num_iteration + 1, 5),
                            xlim_left=0,
                            xlim_right=alg_param.num_iteration,
                            ylim_top=1,
                            xlabel='number of iterations',
                            ylabel='MSE')
    fig.savefig('figure/MSE_vs_iteration_N(sparse).pdf')


if __name__ == '__main__':
    main()
