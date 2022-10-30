#
# MSE performance versus Delta (measurement ratio) in sparse vector reconstruction (Fig. 5)
#

import numpy as np
from tqdm import tqdm
from my_module import cs, cgmt, my_plot


def main():
    # problem settings
    array_delta_prediction = np.linspace(0.3, 0.9, 25)  # measurement ratio for prediction
    array_delta_empirical = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # measurement ratio for empirical performance
    prob_param = cs.ProbParam(N=500,  # dimension of the unknown vector
                              distribution='sparse',  # distribution of the unknown vector
                              p0=0.85,  # sparsity of the unknown vector
                              sigma2_v=1e-3)  # noise variance

    # algorithm settings
    alg_param = cs.AlgParam(rho=0.1, 
                            prox=cs.prox_L1, 
                            num_iteration=20)

    num_empirical = 100  # number of samples for the empirical performance evaluation (>= 100 is better)
    sample_size = 100000  # sample size for the prediction (>= 100000 is better)

    # performance prediction
    array_MSE_prediction = np.zeros((alg_param.num_iteration + 1, array_delta_prediction.size))
    array_MSE_opt = np.zeros(array_delta_prediction.size)
    for i, delta in enumerate(tqdm(array_delta_prediction, desc='theoretical prediction', colour='blue', ncols=80)):
        prob_param.delta = delta
        array_MSE_opt[i], alg_param.lmd = cgmt.optimal_performance(prob_param, sample_size=sample_size)
        array_theoretical_performance = cgmt.state_evolution(prob_param, alg_param, sample_size)
        array_MSE_prediction[:, i:(i + 1)] = array_theoretical_performance.reshape((alg_param.num_iteration + 1, 1))

    # empirical performance
    array_MSE_empirical = np.zeros((alg_param.num_iteration + 1, array_delta_empirical.size))
    for i, delta in enumerate(tqdm(array_delta_empirical, desc='empirical performance', colour='blue', ncols=80)):
        prob_param.delta = delta
        _, alg_param.lmd = cgmt.optimal_performance(prob_param, sample_size=sample_size)
        array_empirical_performance = cs.empirical_performance(prob_param, alg_param, num_empirical=num_empirical)
        array_MSE_empirical[:, i:(i + 1)] = array_empirical_performance.reshape((alg_param.num_iteration + 1, 1))

    # plot results
    fig, ax, array_marker, array_color = my_plot.setup()
    line_width = 2
    marker_size = 10

    array_plot_index = np.array([1, 5, 10, 20])
    for i, k in enumerate(array_plot_index):
        ax.plot(array_delta_empirical, array_MSE_empirical[k, :],
                label=rf'empirical (${k=}$)', linestyle='', color=array_color[i], marker=array_marker[i], markersize=marker_size, clip_on=False)
    for i, k in enumerate(array_plot_index):
        if i == 0:
            ax.plot(array_delta_prediction, array_MSE_prediction[k, :],
                    label='prediction', linestyle='-', color=array_color[i], marker='', linewidth=line_width)
        else:
            ax.plot(array_delta_prediction, array_MSE_prediction[k, :],
                    linestyle='-', color=array_color[i], marker='', linewidth=line_width)
    ax.plot(array_delta_prediction, array_MSE_opt,
            label='asymptotic MSE of optimizer', linestyle='--', color='k', linewidth=line_width)

    my_plot.set_ax_property(ax,
                            yscale='log',
                            xlim_left=np.min(array_delta_prediction),
                            xlim_right=np.max(array_delta_prediction),
                            ylim_top=1,
                            xlabel=r'$\Delta$',
                            ylabel='MSE')
    ax.legend(fontsize=16)
    fig.savefig('figure/MSE_vs_Delta(sparse).pdf')


if __name__ == '__main__':
    main()
