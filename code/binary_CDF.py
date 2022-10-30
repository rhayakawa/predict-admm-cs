#
# Comparison between empirical CDF and its prediction in binary vector reconstruction (Fig. 7)
#

import numpy as np
from tqdm import trange
from my_module import cs, cgmt, my_plot


def main():
    # problem settings
    prob_param = cs.ProbParam(N=500,  # dimension of the unknown vector
                              delta=0.8,  # measurement ratio
                              distribution='binary',  # distribution of the unknown vector
                              sigma2_v=1e-3)  # noise variance

    # algorithm settings
    alg_param = cs.AlgParam(rho=0.1,  # parameter of ADMM
                            prox=cs.prox_box,  # proximity operator
                            num_iteration=7)  # number of iterations

    num_empirical = 100  # number of samples for the empirical performance evaluation (>= 100 is better)
    sample_size = 100000  # sample size for the prediction (>= 100000 is better)

    # performance prediction
    dict_S = {}
    dict_Z = {}
    dict_W = {}
    X, H, Z, W = cgmt.make_variables(prob_param, sample_size=sample_size)
    for index in trange(1, alg_param.num_iteration + 1, desc='state evolution', colour='blue', ncols=80):
        alpha_opt, beta_opt = cgmt.optimal_alpha_beta(prob_param.delta, prob_param.sigma2_v, alg_param.rho, X, H, Z, W)
        S = cgmt.update_s(alpha_opt, beta_opt, prob_param.delta, alg_param.rho, X, H, Z, W)
        Z = cgmt.update_z(S, W, alg_param.lmd, alg_param.rho, alg_param.prox)
        W = cgmt.update_w(W, S, Z)
        dict_S[index] = S
        dict_Z[index] = Z
        dict_W[index] = W

    # empirical performance
    tensor_S_empirical = np.zeros((prob_param.N, num_empirical, alg_param.num_iteration + 1))
    for index_empirical in trange(num_empirical, desc='empirical performance', colour='blue', ncols=80):
        y, A, _ = cs.make_problem(prob_param)
        Ay = A.T @ y
        inv_matrix = cs.inverse_matrix(A, alg_param.rho)
        z = np.zeros((prob_param.N, 1))
        w = np.zeros((prob_param.N, 1))
        for index_iteration in range(1, alg_param.num_iteration + 1):
            s = inv_matrix @ (Ay + alg_param.rho * (z - w))
            z = alg_param.prox(s + w, alg_param.lmd / alg_param.rho)
            w = w + s - z
            tensor_S_empirical[:, index_empirical:(index_empirical + 1), index_iteration] = s

    # plot results
    array_plot_index = np.array([1, 3, 7])
    fig, ax, _, _ = my_plot.setup(1, array_plot_index.size, figsize=(21, 6))
    line_width = 2

    num_bins = 1000
    num_bins_empirical = 60
    array_ordinal = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th', 'th', 'th', 'th', 'th', 'th', 'th']
    for i, index_iteration in enumerate(array_plot_index):
        density, bin_edges = np.histogram(dict_S[index_iteration],
                                          bins=num_bins, range=(-3, 3))
        cumulative = np.cumsum(density)
        ax[i].plot(bin_edges[0:num_bins], cumulative / sample_size,
                   label='prediction', linestyle='--', linewidth=line_width)
        density_empirical, bin_edges_empirical = np.histogram(tensor_S_empirical[:, :, index_iteration],
                                                              bins=num_bins_empirical, range=(-3, 3))
        cumulative_empirical = np.cumsum(density_empirical)
        ax[i].step(bin_edges_empirical[0:num_bins_empirical], cumulative_empirical / (prob_param.N * num_empirical),
                   where='post', label='empirical', linestyle='-', linewidth=line_width)

        my_plot.set_ax_property(ax[i],
                                xlim_left=-2,
                                xlim_right=2,
                                ylim_top=1,
                                ylim_bottom=0,
                                xlabel=r'$s$',
                                ylabel='cumulative distribution',
                                title=rf'{index_iteration}{array_ordinal[index_iteration]} iteration',
                                fontsize=20)
    fig.savefig('figure/CDF(binary).pdf')


if __name__ == '__main__':
    main()
