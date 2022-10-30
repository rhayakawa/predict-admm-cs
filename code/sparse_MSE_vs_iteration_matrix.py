#
# MSE performance for different measurement matrix in sparse vector reconstruction (Fig. 3)
#

from my_module import cs, cgmt, my_plot


def main():
    # problem settings
    prob_param = cs.ProbParam(N=500,  # dimension of the unknown vector
                              delta=0.5,  # measurement ratio
                              distribution='sparse',  # distribution of the unknown vector
                              p0=0.9,  # sparsity of the unknown vector
                              sigma2_v=1e-3) # noise variance

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
    array_MSE_empirical_Gaussian = cs.empirical_performance(prob_param, alg_param, num_empirical=num_empirical, leave=True)
    prob_param.matrix_structure = 'Bernoulli'
    array_MSE_empirical_Bernoulli = cs.empirical_performance(prob_param, alg_param, num_empirical=num_empirical, leave=True)

    # plot results
    fig, ax, array_marker, _ = my_plot.setup()
    line_width = 2
    marker_size = 10

    ax.plot(range(alg_param.num_iteration + 1), array_MSE_empirical_Gaussian,
            label='Gaussian', linestyle='', marker=array_marker[0], markersize=marker_size)
    ax.plot(range(alg_param.num_iteration + 1), array_MSE_empirical_Bernoulli,
            label='Bernoulli', linestyle='', marker=array_marker[1], markersize=marker_size)
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
                            ylim_bottom=1e-3,
                            xlabel='number of iterations',
                            ylabel='MSE')
    fig.savefig('figure/MSE_vs_iteration_matrix(sparse).pdf')


if __name__ == '__main__':
    main()
