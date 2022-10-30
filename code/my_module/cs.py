import numpy as np
from tqdm import trange


class ProbParam:
    """
    parameters of the problem

    N: int
        dimension of the unknown vector
    delta: float
        measurement ratio
    distribution: string
        distribution of the unknown vector
            'sparse' - Bernoulli-Gaussian distribution
            'binary' - binary distribution on {1, -1}
    p0: float
        sparsity of the unknown vector (probability of zero) when distribution == 'sparse'
    sigma2_v: float
        noise variance
    matrix_structure: string, optional
        distribution of the i.i.d. measurement matrix, by default 'Gaussian'
            'Gaussian' - Gaussian distribution
            'Bernoulli' - Bernoulli distribution on {1/sqrt(N), -1/sqrt(N)}
    """
    def __init__(self, N=0, delta=0, distribution='', p0=0, sigma2_v=0, matrix_structure='Gaussian'):
        self.N = N
        self.delta = delta
        self.distribution = distribution
        self.p0 = p0
        self.sigma2_v = sigma2_v
        self.matrix_structure = matrix_structure


class AlgParam:
    """
    parameters of the reconstruction algorithm with ADMM

    lmd: float
        regularization parameter
    rho: float
        parameter of ADMM
    prox: function
        proximity operator
    num_iteration: int, optinoal
        number of iterations, by default 30
    """
    def __init__(self, lmd=0, rho=0, prox=0, num_iteration=30):
        self.lmd = lmd
        self.rho = rho
        self.prox = prox
        self.num_iteration = num_iteration


def empirical_performance(prob_param, alg_param, num_empirical=100, measure='MSE', leave=False):
    """
    compute empirical performance of the reconstruction

    Parameters
    ----------
    prob_param :
        parameters of the problem
    alg_param :
        parameters of the algorithm
    num_empirical : int, optinoal
        number of problem samples, by default 100
    measure : string, optional
        performance measure, by default 'MSE'
            'MSE' - mean squared error
            'SER' - symbol error rate for binary vector
    leave: bool, optional
        option for tqdm (If leave is True, the progress bar remains.), by default False

    Returns
    -------
    array_average_performance : array
        array of average performance (1D array)
    """

    array_sum_performance = np.zeros(alg_param.num_iteration + 1)
    for _ in trange(num_empirical, desc='empirical reconstruction', leave=leave, colour='green', ncols=80):
        y, A, x = make_problem(prob_param)
        _, array_performance = reconstruct(y, A, alg_param, x, measure)
        array_sum_performance += array_performance
    array_average_performance = array_sum_performance / num_empirical

    return array_average_performance


def make_problem(prob_param):
    """
    make an instance of a compressed sensing problem

    Parameters
    ----------
    prob_param :
        parameters of the problem

    Returns
    -------
    y : array
        measurement vector (1D array)
    A : array
        measurement matrix (2D array)
    x : array
        unknown vector (1D array)
    """

    N = prob_param.N
    M = int(N * prob_param.delta)
    distribution = prob_param.distribution
    sigma2_v = prob_param.sigma2_v
    matrix_structure = prob_param.matrix_structure

    rng = np.random.default_rng()
    # unknown vector
    if distribution == 'sparse':
        x = rng.standard_normal((N, 1))
        nonzero_index = (rng.random((N, 1)) > prob_param.p0)
        x = x * nonzero_index
    elif distribution == 'binary':
        x = 2 * rng.integers(2, size=(N, 1)) - np.ones((N, 1))
    # measurement matrix
    if matrix_structure == 'Gaussian':
        A = rng.standard_normal((M, N)) / np.sqrt(N)
    elif matrix_structure == 'Bernoulli':
        A = (2 * rng.integers(2, size=(M, N)) - np.ones((M, N))) / np.sqrt(N)
    # additive noise vector
    v = rng.standard_normal((M, 1)) * np.sqrt(sigma2_v)
    # measurement vector
    y = A @ x + v

    return y, A, x


def reconstruct(y, A, alg_param, x_true, measure='MSE'):
    """
    reconstruct the unknown vector

    Parameters
    ----------
    y : array
        measurement vector (1D array)
    A : array
        measurement matrix (2D array)
    alg_param :
        parameters of the algorithm
    x_true : array
        true unknown vector used for the evaluation of performance (1D array)
    measure : string, optional
        performance measure, by default 'MSE'
            'MSE' - mean squared error
            'SER' - symbol error rate for binary vector

    Returns
    -------
    s : array
        estimate of unknown vector (1D array)
    array_performance : array
        array of performance (1D array)
    """

    lmd = alg_param.lmd
    rho = alg_param.rho
    prox = alg_param.prox
    num_iteration = alg_param.num_iteration

    _, N = A.shape
    Ay = A.T @ y
    inv_matrix = inverse_matrix(A, rho)
    s = np.zeros((N, 1))
    z = np.zeros((N, 1))
    w = np.zeros((N, 1))
    array_performance = np.zeros(num_iteration + 1)
    array_performance[0] = get_error(s, x_true, measure)
    for index in range(1, num_iteration + 1):
        s = inv_matrix @ (Ay + rho * (z - w))
        z = prox(s + w, lmd / rho)
        w = w + s - z
        array_performance[index] = get_error(s, x_true, measure)

    return s, array_performance


def inverse_matrix(A, rho):
    """
    compute the inverse matrix used in ADMM

    Parameters
    ----------
    A : array
        measurement matrix (2D array)
    rho : float
        parameter for ADMM

    Returns
    -------
    inv_matrix : array
        inverse matrix (2D array)
    """

    _, N = A.shape
    inv_matrix = np.linalg.inv(A.T @ A + rho * np.eye(N))

    return inv_matrix


def prox_L1(u, gamma):
    """
    soft thresholding function (proximity operator of L1 norm)

    Parameters
    ----------
    u : array
        input vector (1D array)
    gamma : float
        threshold parameter

    Returns
    -------
    prox_u : array
        output vector (1D array)
    """

    prox_u = np.sign(u) * np.maximum(np.abs(u) - gamma, 0)

    return prox_u


def prox_box(u, gamma):
    """
    proximity operator of indicator function for box constraint [-1, 1]

    Parameters
    ----------
    u : array
        input vector (1D array)
    gamma : float
        threshold parameter (not used for box constraint)

    Returns
    -------
    prox_u : array
        output vector (1D array)
    """

    prox_u = np.minimum(np.maximum(u, -1), 1)

    return prox_u


def get_error(x_est, x_true, measure='MSE'):
    """
    compute error performance

    Parameters
    ----------
    x_est : array
        estimate of the unknown vector (1D array)
    x_true : _type_
        true unknown vector (1D array)
    measure : str, optional
        performance measure, by default 'MSE'
            'MSE' - mean squared error
            'SER' - symbol error rate for binary vector

    Returns
    -------
    error of the estimate of the unknown vector
    """
    if measure == 'MSE':
        return (np.linalg.norm(x_est - x_true) ** 2) / x_est.size
    elif measure == 'SER':
        return np.sum((x_est >= 0) != (x_true >= 0)) / x_est.size
