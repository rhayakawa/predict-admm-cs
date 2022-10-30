#
# MSE performance for different rho (parameter of ADMM) in sparse vector reconstruction (Fig. 4)
#

import numpy as np
from tqdm import tqdm
from my_module import cs, cgmt, my_plot


def main():
    # problem settings
    prob_param = cs.ProbParam(N=500,  # dimension of the unknown vector
                              delta=0.8,  # measurement ratio
                              distribution='sparse',  # distribution of the unknown vector
                              p0=0.9,  # sparsity of the unknown vector
                              sigma2_v=5e-3)  # noise variance

    # algorithm settings
    array_rho = np.array([0.05, 0.2, 0.5])  # parameter of ADMM
    alg_param = cs.AlgParam(prox=cs.prox_L1,  # proximity operator
                            num_iteration=20)  # number of iterations

    num_empirical = 100  # number of samples for the empirical performance evaluation (>= 100 is better)
    sample_size = 100000   # sample size for the prediction (>= 100000 is better)

    # optimal MSE and regularization parameter
    MSE_opt, alg_param.lmd = cgmt.optimal_performance(prob_param, sample_size=sample_size)

    dict_array_MSE_prediction = {}
    dict_array_MSE_empirical = {}
    for rho in tqdm(array_rho, desc='empirical performance', colour='blue', ncols=80):
        alg_param.rho = rho
        # performance prediction
        dict_array_MSE_prediction[rho] = cgmt.state_evolution(prob_param, alg_param, sample_size=sample_size)
        # empirical performance
        dict_array_MSE_empirical[rho] = cs.empirical_performance(prob_param, alg_param, num_empirical=num_empirical)

    # plot results
    fig, ax, array_marker, array_color = my_plot.setup()
    line_width = 2
    marker_size = 10

    for i, rho in enumerate(array_rho):
        ax.plot(range(alg_param.num_iteration + 1), dict_array_MSE_empirical[rho],
                label=rf'empirical ($\rho = {rho}$)', linestyle='', color=array_color[i], marker=array_marker[i], markersize=marker_size)
    for i, rho in enumerate(array_rho):
        if i == 0:
            ax.plot(range(alg_param.num_iteration + 1), dict_array_MSE_prediction[rho],
                    label='prediction', linestyle='-', color=array_color[i], linewidth=line_width)
        else:
            ax.plot(range(alg_param.num_iteration + 1), dict_array_MSE_prediction[rho],
                    linestyle='-', color=array_color[i], linewidth=line_width)
    ax.hlines(y=MSE_opt, xmin=0, xmax=alg_param.num_iteration,
              label='asymptotic MSE of optimizer', linestyle='--', color='k', linewidth=line_width)

    my_plot.set_ax_property(ax,
                            yscale='log',
                            xticks=range(0, alg_param.num_iteration + 1, 5),
                            xlim_left=0,
                            xlim_right=alg_param.num_iteration,
                            ylim_top=1,
                            ylim_bottom=1e-3,
                            xlabel='number of iterations',
                            ylabel='MSE')
    fig.savefig('figure/MSE_vs_iteration_rho(sparse).pdf')


if __name__ == '__main__':
    main()
