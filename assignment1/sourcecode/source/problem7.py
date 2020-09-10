import source.problem6 as libgumbel
import numpy as np
from scipy.spatial.distance import euclidean
from statistics import mean, stdev


def gaussian(data, mu, sigma):
    return list(map(lambda x: (np.exp(-np.square(x - mu) / (2 * np.square(sigma)))) / (sigma * np.sqrt(2 * np.pi)), data))


def gumbel(data, alpha, beta):
    x_minus_data = libgumbel.x_minus_alpha(data, alpha)
    exps = libgumbel.exponential_function(x_minus_data, beta)
    var1 = list(map(lambda x: np.exp(-x), exps))
    return np.divide(np.multiply(exps, var1), beta)


def Pyi(data, primary_wk, primary_pdf, secondary_wk, secondary_pdf, primary_param1, primary_param2, secondary_param1,
        secondary_param2):
    primary_distribution = primary_pdf(data, primary_param1, primary_param2)
    w_pyi_primary = np.multiply(primary_distribution, primary_wk)
    secondary_distribution = secondary_pdf(data, secondary_param1, secondary_param2)
    w_pyi_secondary = np.multiply(secondary_distribution, secondary_wk)
    denom = np.add(w_pyi_primary, w_pyi_secondary)
    return np.divide(w_pyi_primary, denom)


def gaussian_data_generator(sample_size, mu, sigma):
    return [np.random.normal(mu, sigma, sample_size) for _ in range(0, 10)]


def em(data, initial_w_gauss, initial_w_gumbel, initial_mu, initial_sigma, initial_alpha, initial_beta):
    t = 0
    t_max = 1000
    while t < t_max:
        common_denom = np.add(np.multiply(initial_w_gauss, gaussian(data, initial_mu, initial_sigma)),
                              np.multiply(initial_w_gumbel, gumbel(data, initial_alpha, initial_beta)))

        # next_Pyi_Gaussian = Pyi(data, initial_w_gauss, gaussian, initial_w_gumbel, gumbel, initial_mu, initial_sigma,
        #                         initial_alpha, initial_beta)
        next_Pyi_Gaussian = np.divide(np.multiply(initial_w_gauss, gaussian(data, initial_mu, initial_sigma)),
                                      common_denom)

        # next_Pyi_Gumbel = Pyi(data, initial_w_gumbel, gumbel, initial_w_gauss, gaussian, initial_alpha, initial_beta,
        #                       initial_mu, initial_sigma)

        next_Pyi_Gumbel = np.divide(np.multiply(initial_w_gumbel, gumbel(data, initial_alpha, initial_beta)),
                                    common_denom)

        sum_pyi_gaussian = sum(next_Pyi_Gaussian)
        sum_pyi_gumbel = sum(next_Pyi_Gumbel)

        next_w_gauss = sum_pyi_gaussian / len(data)
        next_w_gumbel = sum_pyi_gumbel / len(data)

        next_mu = sum(np.multiply(data, next_Pyi_Gaussian)) / sum_pyi_gaussian
        x_minus_mu_square = list(map(lambda x: x ** 2, libgumbel.x_minus_alpha(data, initial_mu)))
        next_sigma = np.sqrt(sum(np.multiply(x_minus_mu_square, next_Pyi_Gaussian)) / sum_pyi_gaussian)

        x_minus_alpha_data = libgumbel.x_minus_alpha(data, initial_alpha)
        first_derivatives_gumbel = np.matrix(
            [libgumbel.alpha_update_rule(x_minus_alpha_data, initial_beta, next_Pyi_Gumbel),
             libgumbel.beta_update_rule(x_minus_alpha_data, sum(np.multiply(data, next_Pyi_Gumbel)), initial_alpha,
                                        initial_beta, next_Pyi_Gumbel)])

        alpha_beta_updates = np.matmul(
            libgumbel.hessian_inverse(x_minus_alpha_data, initial_alpha, initial_beta, data, next_Pyi_Gumbel),
            np.transpose(first_derivatives_gumbel))

        next_alpha = initial_alpha - alpha_beta_updates.item(0)
        next_beta = initial_beta - alpha_beta_updates.item(1)

        # if t % 1 == 0:
        #     print("pyi for gauss = {} and pyi for gumbel ={}".format(sum_pyi_gaussian, sum_pyi_gumbel))
        #     print("Iteration t= {0}/{1}: old and new mu = {2},{3}. old and new sigma = {4},{5} old "
        #           "and new w_gauss = {6},{7}. old and new w_gumbel = {8},{9}. old and new alpha = {10},{11}."
        #           "old and new beta = {12},{13}".format(t, t_max, initial_mu, next_mu, initial_sigma, next_sigma,
        #                                                 initial_w_gauss, next_w_gauss, initial_w_gumbel, next_w_gumbel,
        #                                                 initial_alpha, next_alpha, initial_beta, next_beta))

        old_values = [initial_w_gauss, initial_w_gumbel, initial_mu, initial_sigma, initial_alpha, initial_beta]
        new_values = [next_w_gauss, next_w_gumbel, next_mu, next_sigma, next_alpha, next_beta]

        diff = euclidean(new_values, old_values)

        if diff <= delta:
            break

        initial_w_gauss = next_w_gauss
        initial_w_gumbel = next_w_gumbel
        initial_mu = next_mu
        initial_sigma = next_sigma
        initial_alpha = next_alpha
        initial_beta = next_beta

        t += 1

    return next_w_gauss, next_w_gumbel, next_mu, next_sigma, next_alpha, next_beta


w_gauss = 0.4
w_gumbel = 0.6
sample_mu = 6
sample_sigma = 1
sample_alpha = 2
sample_beta = 1
delta = 0.0001

for n in [100, 1000, 10000]:
    np.random.seed(1)
    gaussian_data = gaussian_data_generator(int(n * w_gauss), sample_mu, sample_sigma)
    gumbel_data = libgumbel.gumbel_data_generator(int(n * w_gumbel), sample_alpha, sample_beta)
    datasets = [np.concatenate((data1, data2), axis=None) for data1, data2 in zip(gaussian_data, gumbel_data)]
    first_w_gauss = 0.5
    first_w_gumbel = 0.5
    first_mu = 3
    first_sigma = 0.5
    first_alpha = 1
    first_beta = 0.5
    estimated_parameters = []
    for i, dataset in enumerate(datasets):
        # first_mu = mean(dataset)
        # first_sigma = np.var(dataset)
        # first_alpha = mean(dataset)/2
        # first_beta = 2*np.var(dataset)
        #np.random.shuffle(dataset)
        #print("Dataset {}".format(dataset))
        estimated_parameters.append(
            em(dataset, first_w_gauss, first_w_gumbel, first_mu, first_sigma, first_alpha, first_beta))
        print("Done dataset {}/10".format(i + 1))

    final_w_gauss, final_w_gumbel, final_mu, final_sigma, final_alpha, final_beta = zip(*estimated_parameters)

    print("For n = {} Mean w_gauss = {} and Mean w_gumbel = {}".format(n, mean(final_w_gauss), mean(final_w_gumbel)))
    print("For n = {} Standard deviation of w_gauss ={} and w_gumbel = {}".format(n, stdev(final_w_gauss),
                                                                                  stdev(final_w_gumbel)))
    print("For n = {} Mean mu = {} and Mean sigma = {}".format(n, mean(final_mu), mean(final_sigma)))
    print("For n = {} Standard deviation of mu ={} and sigma = {}".format(n, stdev(final_mu), stdev(final_sigma)))
    print("For n = {} Mean alpha = {} and Mean beta = {}".format(n, mean(final_alpha), mean(final_beta)))
    print("For n = {} Standard deviation of alpha ={} and beta = {}".format(n, stdev(final_alpha), stdev(final_beta)))
