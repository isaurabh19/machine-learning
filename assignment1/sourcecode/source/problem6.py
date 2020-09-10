import numpy as np
import math
from statistics import mean, stdev


def exponential_function(data, beta):
    data_new = np.divide(data, beta)
    return list(map(lambda x: np.exp(-x), data_new))


def x_minus_alpha(data, alpha):
    return list(map(lambda x: x - alpha, data))


def gumbel_data_generator(num, alpha=2, beta=3):
    return [np.random.gumbel(alpha, beta, num) for _ in range(0, 10)]


def alpha_update_rule(x_minus_alpha_data, beta, Pyi):
    exps = exponential_function(x_minus_alpha_data, beta)
    return (sum(Pyi) - sum(np.multiply(exps, Pyi))) / beta


def beta_update_rule(x_minus_alpha_data, sum_og_data, alpha, beta, Pyi):
    exps = exponential_function(x_minus_alpha_data, beta)
    return (-sum(np.multiply(np.multiply(exps, x_minus_alpha_data), Pyi)) + sum_og_data - (
            (alpha + beta) * sum(Pyi))) / (
                   beta * beta)


def isParamConverged(param0, param1, diff=0.0001):
    return math.fabs(param1 - param0) <= diff


def hessian_inverse(x_minus_alpha_data, alpha, beta, og_data, pyi):
    sum_pyi = sum(pyi)
    exps = exponential_function(x_minus_alpha_data, beta)
    ele1 = np.divide(-sum(np.multiply(exps, pyi)), (beta ** 2))
    temp1 = sum(np.multiply(np.multiply(exps, x_minus_alpha(x_minus_alpha_data, beta)), pyi))
    ele2 = (-sum_pyi * beta - temp1) / (beta ** 3)
    temp2 = np.multiply(pyi, np.multiply(x_minus_alpha_data, exps))
    temp3 = np.multiply(temp2, x_minus_alpha(x_minus_alpha_data, (2.0 * beta)))
    temp4 = np.add(temp3, np.multiply(2 * beta, np.multiply(og_data, pyi)))
    ele4 = (2 * sum_pyi * alpha * beta + sum_pyi * (beta ** 2) - sum(temp4)) / (beta ** 4)
    hessian = np.matrix([[ele1, ele2], [ele2, ele4]])
    return np.linalg.inv(hessian)


def converge(initial_alpha, initial_beta, data, len_data):
    isAlphaConverged = False
    isBetaConverged = False
    sum_of_data = sum(data)
    t = 0
    max_iterations = 10000
    while t < max_iterations:
        x_minus_alpha_data = x_minus_alpha(data, initial_alpha)
        first_derivatives = np.matrix([alpha_update_rule(x_minus_alpha_data, initial_beta, [1] * len_data),
                                       beta_update_rule(x_minus_alpha_data, sum_of_data, initial_alpha,
                                                        initial_beta, [1] * len_data)])

        alpha_beta_updates = np.matmul(
            hessian_inverse(x_minus_alpha_data, initial_alpha, initial_beta, data, [1] * len_data),
            np.transpose(first_derivatives))

        if not isAlphaConverged:
            alpha_next = initial_alpha - alpha_beta_updates.item(0)
        if not isBetaConverged:
            beta_next = initial_beta - alpha_beta_updates.item(1)

        isAlphaConverged = isParamConverged(initial_alpha, alpha_next)
        isBetaConverged = isParamConverged(initial_beta, beta_next)

        if isAlphaConverged and isBetaConverged:
            break

        # if t % 100 == 0:
        #     print("Iteration {0}/{7}: alpha-{0} = {1} | alpha-{0}+1 = {2} \t beta-{0} = {4} | beta-{0}+1 = {5} "
        #           "\t alpha-delta = {3} | beta-delta = {6}\n"
        #           .format(t, initial_alpha, alpha_next, math.fabs(alpha_next - initial_alpha), initial_beta, beta_next,
        #                   math.fabs(beta_next - initial_beta), max_iterations))

        initial_alpha = alpha_next
        initial_beta = beta_next
        t += 1

    return alpha_next, beta_next


def alpha_mle(beta, og_data):
    exps = exponential_function(og_data, beta)
    next = beta * (math.log(len(og_data)) - math.log(sum(exps)))
    return next


def beta_mle(alpha, beta, x_minus_alpha_data, og_data):
    exps = exponential_function(x_minus_alpha_data, beta)
    next = mean(og_data) - alpha - (sum(np.multiply(exps, x_minus_alpha_data)) / len(og_data))
    return next


for n in [100,1000,10000]:
    print("N = {}".format(n))
    np.random.seed(1)
    datasets = gumbel_data_generator(n)

    alpha_start = 1.3  # round(mean(datasets[0]), 3)
    beta_start = 1.5  # round(stdev(datasets[0]), 3)
    estimated_params = []
    for dataset in datasets:
        # alpha_start = round(mean(dataset), 3)
        # beta_start = np.var(dataset, axis=None)
        estimated_params.append(converge(alpha_start, beta_start, dataset, len(dataset)))

    alphas, betas = zip(*estimated_params)

    print("For n = {} Mean alpha = {} and Mean beta = {}".format(n, mean(alphas), mean(betas)))
    print("For n = {} Standard deviation of alpha ={} and beta = {}".format(n, stdev(alphas), stdev(betas)))
