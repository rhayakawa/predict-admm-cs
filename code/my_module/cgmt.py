import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from tqdm import trange
from my_module import cs


def state_evolution(prob_param, alg_param, sample_size=100000, measure='MSE', leave=False):
    """
    get the evolution of the asymptotic performance in ADMM

    Parameters
    ----------
    prob_param :
        parameters of the problem
    alg_param :
        parameters of the algorithm
    sample_size : int, optional
        sample size for random variables, by default 100000
    measure : string, optional
        performance measure, by default 'MSE'
            'MSE' - mean squared error
            'SER' - symbol error rate
    leave: bool, optional
        option for tqdm (If leave is True, the progress bar remains.), by default False

    Returns
    -------
    array_performance : array
        array for performance prediction (1D array)
    """

    # problem parameter
    delta = prob_param.delta
    sigma2_v = prob_param.sigma2_v

    # algorithm parameter
    lmd = alg_param.lmd
    rho = alg_param.rho
    prox = alg_param.prox
    num_iteration = alg_param.num_iteration

    X, H, Z, W = make_variables(prob_param, sample_size)

    array_performance = np.zeros(num_iteration + 1)
    if measure == 'MSE':
        array_performance[0] = (np.linalg.norm(X) ** 2) / sample_size
    elif measure == 'SER':
        array_performance[0] = 1 / 2
    for index in trange(1, num_iteration + 1, desc='state evolution', leave=leave, colour='green', ncols=80):
        alpha_opt, beta_opt = optimal_alpha_beta(delta, sigma2_v, rho, X, H, Z, W)
        S = update_s(alpha_opt, beta_opt, delta, rho, X, H, Z, W)
        Z = update_z(S, W, lmd, rho, prox)
        W = update_w(W, S, Z)
        if measure == 'MSE':
            array_performance[index] = (alpha_opt ** 2) - sigma2_v
        elif measure == 'SER':
            array_performance[index] = np.linalg.norm(np.sign(S[:, 0]) - X[:, 0], ord=0) / sample_size

    return array_performance


def make_variables(prob_param, sample_size):
    """
    make random variables used for the evaluation of the expectation

    Parameters
    ----------
    prob_param :
        parameters of the problem
    sample_size : int
        sample size

    Returns
    -------
    X : array
        Bernoulli-Gaussian random variables (1D array)
    H : array
        Gaussian random variables (1D array)
    Z : array
        initial zero vector (1D array)
    W : array
        initial zero vector (1D array)
    """

    rng = np.random.default_rng(0)

    distribution = prob_param.distribution
    if distribution == 'sparse':
        X = rng.standard_normal((sample_size, 1))
        nonzero_index = (rng.random((sample_size, 1)) > prob_param.p0)
        X = X * nonzero_index
    elif distribution == 'binary':
        X = 2 * rng.integers(2, size=(sample_size, 1)) - np.ones((sample_size, 1))
    H = rng.standard_normal((sample_size, 1))
    Z = np.zeros((sample_size, 1))
    W = np.zeros((sample_size, 1))

    return X, H, Z, W


def optimal_alpha_beta(delta, sigma2_v, rho, X, H, Z, W, alpha_left=0, alpha_right=10, beta_left=0, beta_right=10):
    """
    compute optimal values of alpha and beta

    Parameters
    ----------
    delta : float
        measurement ratio
    sigma2_v : float
        noise variance
    rho : float
        parameter of ADMM
    X : array
        Bernoulli-Gaussian random variables (1D array)
    H : array
        Gaussian random variables (1D array)
    Z : array
        random variables (1D array)
    W : array
        random variables (1D array)
    alpha_left : float, optional
        left end of the search space for alpha, by default 0
    alpha_right : float, optional
        right end of the search space for alpha, by default 10
    beta_left : float, optional
        left end of the search space for beta, by default 0
    beta_right : float, optional
        right end of the search space for beta, by default 10

    Returns
    -------
    alpha_opt : float
        optimal value of alpha
    beta_opt : float
        optimal value of beta
    """

    result_opt = minimize_scalar(lambda alpha: optimize_beta(alpha, delta, sigma2_v, rho, X, H, Z, W, beta_left, beta_right),
                                 bounds=(alpha_left, alpha_right), method='bounded', options={'disp': 1})
    alpha_opt = result_opt.x
    result_opt2 = minimize_scalar(lambda beta: -obj_func_so(alpha_opt, beta, delta, sigma2_v, rho, X, H, Z, W),
                                  bounds=(beta_left, beta_right), method='bounded', options={'disp': 1})
    beta_opt = result_opt2.x

    return alpha_opt, beta_opt


def optimize_beta(alpha, delta, sigma2_v, rho, X, H, Z, W, beta_left, beta_right):
    """
    optimization of scalar optimization problem over beta

    Parameters
    ----------
    alpha : float
        variable to be optimized
    delta : float
        measurement ratio
    sigma2_v : float
        noise variance
    rho : float
        parameter of ADMM
    X : array
        Bernoulli-Gaussian random variables (1D array)
    H : array
        Gaussian random variables (1D array)
    Z : array
        random variables (1D array)
    W : array
        random variables (1D array)
    beta_left : float
        left end of the search space for beta
    beta_right : float
        right end of the search space for beta

    Returns
    -------
    -result.fun : float
        optimal value of the objective function
    """

    result = minimize_scalar(lambda beta: -obj_func_so(alpha, beta, delta, sigma2_v, rho, X, H, Z, W),
                             bounds=(beta_left, beta_right), method='bounded', options={'disp': 1})

    return -result.fun


def obj_func_so(alpha, beta, delta, sigma2_v, rho, X, H, Z, W):
    """
    objective function of the scalar optimization problem corresponding to the subproblem in ADMM

    Parameters
    ----------
    alpha : float
        variable to be optimized
    beta : float
        variable to be optimized
    delta : float
        measurement ratio
    sigma2_v : float
        noise variance
    rho : float
        parameter of ADMM
    X : array
        Bernoulli-Gaussian random variables (1D array)
    H : array
        Gaussian random variables (1D array)
    Z : array
        random variables (1D array)
    W : array
        random variables (1D array)

    Returns
    -------
    objective_value : float
        output of the objective function
    """

    objective_value = \
        alpha * beta * np.sqrt(delta) / 2 \
        + beta * sigma2_v * np.sqrt(delta) / (2 * alpha) \
        - (beta ** 2) / 2 \
        + expectation(alpha, beta, delta, rho, X, H, Z, W)

    return objective_value


def expectation(alpha, beta, delta, rho, X, H, Z, W):
    S = update_s(alpha, beta, delta, rho, X, H, Z, W)
    J = (beta * np.sqrt(delta)) / (2 * alpha) * ((S - X) ** 2) \
        - beta * H * (S - X) \
        + (rho / 2) * ((S - Z + W) ** 2)

    return np.mean(J)


def update_s(alpha, beta, delta, rho, X, H, Z, W):
    return 1 / ((beta * np.sqrt(delta) / alpha) + rho) \
           * ((beta * np.sqrt(delta) / alpha) * (X + (alpha / np.sqrt(delta)) * H) \
           + rho * (Z - W))


def update_z(S, W, lmd, rho, prox):
    return prox(S + W, lmd / rho)


def update_w(W, S, Z):
    return W + S - Z


def optimal_performance(prob_param, sample_size=100000, measure='MSE', lmd_left=0, lmd_right=1):
    """
    get optimal performance and the corresponding regularization parameter lambda

    Parameters
    ----------
    prob_param :
        parameters of the problem
    sample_size : int, optional
        sample size of the random variables, by default 100000
    measure : string, optional
        performance measure, by default 'MSE'
            'MSE' - mean squared error
            'SER' - symbol error rate
    lmd_left : float, optional
        left end of the search space, by default 0
    lmd_right : float, optional
        right end of the search space, by default 1

    Returns
    -------
    performance_opt : float
        optimal asymptotic performance
    lmd_opt : float
        optimal value of lambda
    """

    X, H, _, _ = make_variables(prob_param, sample_size=sample_size)
    result_opt = minimize_scalar(lambda lmd: final_performance(lmd, prob_param, X, H, measure=measure),
                                 bounds=(lmd_left, lmd_right), method='bounded', options={'disp': 1})
    lmd_opt = result_opt.x
    performance_opt = result_opt.fun

    return performance_opt, lmd_opt


def final_performance(lmd, prob_param, X, H, measure='MSE', alpha_left=0, alpha_right=10, beta_left=0, beta_right=10):
    """
    get asymptotic performance of the original optimization problem

    Parameters
    ----------
    lmd : float
        regularization parameter
    prob_param :
        parameters of the problem
    X : array
        Bernoulli-Gaussian random variables (1D array)
    H : array
        Gaussian random variables (1D array)
    measure : string, optional
        performance measure, by default 'MSE'
            'MSE' - mean squared error
            'SER' - symbol error rate
    alpha_left : float, optional
        left end of the search space for alpha, by default 0
    alpha_right : float, optional
        right end of the search space for alpha, by default 10
    beta_left : float, optional
        left end of the search space for beta, by default 0
    beta_right : float, optional
        right end of the search space for beta, by default 10

    Returns
    -------
    performance : float
        asymptotic performance
    """

    result_opt = minimize_scalar(lambda alpha: so_with_opt_beta_orig(alpha, lmd, prob_param, X, H, beta_left, beta_right),
                                 bounds=(alpha_left, alpha_right), method='bounded', options={'disp': 1})
    alpha_opt = result_opt.x

    if measure == 'MSE':
        performance = (alpha_opt ** 2) - prob_param.sigma2_v
    elif measure == 'SER':
        performance = 1 - norm.cdf(np.sqrt(prob_param.delta) / alpha_opt)

    return performance


def so_with_opt_beta_orig(alpha, lmd, prob_param, X, H, beta_left, beta_right):
    """
    optimization of scalar optimization problem over beta

    Parameters
    ----------
    alpha : float
        variable to be optimized
    lmd : float
        regularization parameter
    prob_param :
        parameters of the problem
    X : array
        Bernoulli-Gaussian random variables (1D array)
    H : array
        Gaussian random variables (1D array)
    beta_left : float
        left end of the search space for beta
    beta_right : float
        right end of the search space for beta

    Returns
    -------
    -result.fun : float
        optimal value of the objective function
    """

    result = minimize_scalar(lambda beta: -so_orig(alpha, beta, lmd, prob_param, X, H),
                             bounds=(beta_left, beta_right), method='bounded', options={'disp': 1})

    return -result.fun


def so_orig(alpha, beta, lmd, prob_param, X, H):
    """
    objective function of the scalar optimization corresponding to the original problem

    Parameters
    ----------
    alpha : float
        variable to be optimized
    beta : float
        variable to be optimized
    lmd : float
        regularization parameter
    prob_param :
        parameters of the problem
    X : array
        Bernoulli-Gaussian random variables (1D array)
    H : array
        Gaussian random variables (1D array)

    Returns
    -------
    objective_value : float
        output of the objective function
    """

    delta = prob_param.delta
    distribution = prob_param.distribution
    sigma2_v = prob_param.sigma2_v

    objective_value = \
        alpha * beta * np.sqrt(delta) / 2 \
        + beta * sigma2_v * np.sqrt(delta) / (2 * alpha) \
        - beta ** 2 / 2 \
        - alpha * beta / (2 * np.sqrt(delta)) \
        + beta * np.sqrt(delta) / alpha * expectation_env(alpha, beta, delta, lmd, distribution, X, H)

    return objective_value


def expectation_env(alpha, beta, delta, lmd, distribution, X, H):

    X_noisy = X + alpha / np.sqrt(delta) * H
    threshold = alpha * lmd / (beta * np.sqrt(delta))
    if distribution == 'sparse':
        S = cs.prox_L1(X_noisy, threshold)
        J = 1 / 2 * (X_noisy - S) ** 2 + threshold * np.abs(S)
    elif distribution == 'binary':
        S = cs.prox_box(X_noisy, threshold)
        J = 1 / 2 * (X_noisy - S) ** 2

    return np.mean(J)
