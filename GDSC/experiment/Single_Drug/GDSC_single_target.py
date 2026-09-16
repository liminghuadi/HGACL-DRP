# from import_path import *
from GDSC.experiment.model import GModel
from GDSC.experiment.optimizer import Optimizer
from GDSC.experiment.Single_Drug.sampler import Sampler


def mofgcn_single_target(gene, cna, mutation, drug_feature, response_mat, null_mask, target_index,
                         train_index, test_index, evaluate_fun, sigma=2, knn=11, iterates=3, n_hid1=192, n_hid2=36,
                         alpha=5.74, lr=5e-4, epochs=1000, device="cpu", degree=3, attention_dim=32):
    """
    :param gene: cell gene feature, narray
    :param cna: cell cna feature, narray
    :param mutation:cell mutation feature, narray
    :param drug_feature: drug fingerprint feature, narray
    :param response_mat: response matrix, narray
    :param null_mask: null mask of response_mat, narray
    :param target_index: target index in response matrix, int scale
    :param train_index: train index in original matrix, an vector of narray
    :param test_index: test index in original matrix, an vector of narray
    :param evaluate_fun: evaluate function, parameter must be true data, predict data and true mask
    :param sigma: an scale parameter, int or float el.
    :param knn: KNN parameter, int
    :param iterates: iterate parameter, int
    :param n_hid1: the first hiden layer, int
    :param n_hid2: the second hiden layer, int
    :param alpha: a scale parameter
    :param lr: learning rate, float
    :param epochs: apochs, int
    :param device: run device, cpu or cuda:0
    :return: AUC, ACC, F1-score and so on, an scalar, score
    """
    sample = Sampler(response_mat, null_mask, target_index, train_index, test_index)
    model = GModel(adj_mat=sample.train_data, gene=gene, cna=cna, mutation=mutation,
                   sigma=sigma, k=knn, iterates=iterates, feature_drug=drug_feature,
                   n_hid1=n_hid1, n_hid2=n_hid2, alpha=alpha,
                   degree=degree, attention_dim=attention_dim,
                   device=device)
    opt = Optimizer(model, sample.train_data, sample.test_data, sample.test_mask, sample.train_mask, evaluate_fun,
                    lr=lr, epochs=epochs, device=device)
    epoch, true_data, predict_data = opt()
    return epoch, true_data, predict_data
