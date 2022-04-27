import h5py, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader

from dca import data_util
from dca.data_util import CrossValidate
from dca import DynamicalComponentsAnalysis as DCA
from PFPC import PastFutureDataset, train_PFPC, DCA_init, Polynomial_expand
from utils.plotting.fig1 import plot_lorenz_3d
do_dca_init = True
estimator_compress = "infonce_upper"
estimator_predictive = "infonce_lower"
critic = "concat"
baseline = "constant"
deterministic = False
Kernel = None
hidden_dim = 256
beta = 1e-3
beta1 = 1.
beta2 = 0.
num_epochs = 20
num_early_stop = 0
batch_size = 512
num_vis = 500
do_dca_init = True
do_vis_latent_trials = True
device = "cuda:1"
lr = 0.001
mi_params = {'estimator_compress': estimator_compress, "estimator_predictive": estimator_predictive,
                 "critic": critic,
                 "baseline": baseline}
baseline_params = {"hidden_dim": hidden_dim}
linewidth_3d = 0.5


# run analysis
def run_dca_analysis(X, Y, T_pi_vals, dim_vals, num_cv_folds,
                 n_init=1, verbose=False):

    min_std = 1e-6
    good_cols = (X.std(axis=0) > min_std)
    X = X[:, good_cols]
    # loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds, stack=False)
    X_train_dca_dict = dict()
    X_test_dca_dict = dict()
    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        if verbose:
            print("fold", fold_idx + 1, "of", num_cv_folds)

        fold_name = "fold{}".format(fold_idx)
        X_train_dca_dict[fold_name] = dict()
        X_test_dca_dict[fold_name] = dict()
        # mean-center X and Y
        X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
        X_train_ctd = [Xi - X_mean for Xi in X_train]
        X_test_ctd = X_test - X_mean
        # Y_mean = np.concatenate(Y_train).mean(axis=0, keepdims=True)
        # Y_train_ctd = [Yi - Y_mean for Yi in Y_train]
        # Y_test_ctd = Y_test - Y_mean

        # compute cross-cov mats for DCA
        dca_model = DCA(T=np.max(T_pi_vals))
        dca_model.estimate_data_statistics(X_train_ctd)

        # make DCA object

        # loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            dim_name = "dim{}".format(dim)
            X_train_dca_dict[fold_name][dim_name] = dict()
            X_test_dca_dict[fold_name][dim_name] = dict()
            if verbose:
                print("dim", dim_idx + 1, "of", len(dim_vals))

            # loop over T_pi vals
            for T_pi_idx in range(len(T_pi_vals)):
                T_pi = T_pi_vals[T_pi_idx]
                T_pi_name = "T_pi{}".format(T_pi)
                dca_model.fit_projection(d=dim, T=T_pi, n_init=n_init)
                V_dca = dca_model.coef_

                # compute DCA R2 over offsets
                X_train_dca = [np.dot(Xi, V_dca) for Xi in X_train_ctd]
                X_test_dca = np.dot(X_test_ctd, V_dca)

                X_train_dca_dict[fold_name][dim_name][T_pi_name] = X_train_dca
                X_test_dca_dict[fold_name][dim_name][T_pi_name] = X_test_dca

    return X_train_dca_dict, X_test_dca_dict


def run_analysis_pfpc(X, Y, T_pi_vals, dim_vals, num_cv_folds,
                 n_init=1, verbose=False, Kernel=None, xdim=None, beta=1e-3, beta1=1, beta2=0, good_ts=None):
    """

    :param X:
    :param Y:
    :param T_pi_vals: window size for DCA and CPIC
    :param dim_vals: compressed dimension
    :param offset_vals: Temporal offsets for prediction (0 is same-time prediction)
    :param num_cv_folds:
    :param decoding_window: Number of time samples of X to use for predicting Y (should be odd). Centered around
        offset value.
    :param n_init: the number of initialization for DCA
    :param verbose:
    :return:
    """
    min_std = 1e-6
    good_cols = (X.std(axis=0) > min_std)
    X = X[:, good_cols]
    if good_ts is not None:
        X = X[:good_ts]
        Y = Y[:good_ts]

    if Kernel is not None:
        xdim = int(xdim + xdim * (xdim + 1) / 2)  # polynomial

    # loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds, stack=False)

    X_train_pfpc_dict = dict()
    X_test_pfpc_dict = dict()
    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        if verbose:
            print("fold", fold_idx + 1, "of", num_cv_folds)

        if Kernel is not None:
            X_train = [Kernel(Xi) for Xi in X_train]
            X_test = Kernel(X_test)

        fold_name = "fold{}".format(fold_idx)
        X_train_pfpc_dict[fold_name] = dict()
        X_test_pfpc_dict[fold_name] = dict()
        # mean-center X and Y
        X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
        X_train_ctd = [Xi - X_mean for Xi in X_train]
        X_test_ctd = X_test - X_mean

        # loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            dim_name = "dim{}".format(dim)
            X_train_pfpc_dict[fold_name][dim_name] = dict()
            X_test_pfpc_dict[fold_name][dim_name] = dict()
            if verbose:
                print("dim", dim_idx + 1, "of", len(dim_vals))

            # loop over T_pi vals
            for T_pi_idx in range(len(T_pi_vals)):
                T_pi = T_pi_vals[T_pi_idx]
                T_pi_name = "T_pi{}".format(T_pi)
                critic_params = {"x_dim": T_pi * dim, "y_dim": T_pi * dim, "hidden_dim": hidden_dim}
                critic_params_YX = {"x_dim": T_pi * dim, "y_dim": T_pi * xdim, "hidden_dim": hidden_dim}
                # train data
                if do_dca_init:
                    init_weights = DCA_init(np.concatenate(X_train_ctd, axis=0), T=T_pi, d=dim, n_init=n_init)
                else:
                    init_weights = None

                train_data = PastFutureDataset(X_train_ctd, window_size=T_pi)
                train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                PFPC = train_PFPC(beta, xdim, dim, mi_params, critic_params, baseline_params, num_epochs,
                                  train_dataloader, T=T_pi,
                                  signiture="CPIC", deterministic=deterministic, init_weights=init_weights, lr=lr,
                                  num_early_stop=num_early_stop, device=device, beta1=beta1, beta2=beta2,
                                  critic_params_YX=critic_params_YX).to(device)
                # encode train data and test data via CPIC
                X_train_pfpc = [PFPC.encode(torch.from_numpy(Xi).to(torch.float).to(device)) for Xi in X_train_ctd]
                X_train_pfpc = [Xi.cpu().detach().numpy() for Xi in X_train_pfpc]
                X_test_pfpc = PFPC.encode(torch.from_numpy(X_test_ctd).to(torch.float).to(device))
                X_test_pfpc = X_test_pfpc.cpu().detach().numpy()
                X_train_pfpc_dict[fold_name][dim_name][T_pi_name] = X_train_pfpc
                X_test_pfpc_dict[fold_name][dim_name][T_pi_name] = X_test_pfpc
    return X_train_pfpc_dict, X_test_pfpc_dict


def plot_trials(X, num_vis=200, file_name="true_lorenz_dynamics"):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot_lorenz_3d(ax, X[:num_vis], linewidth_3d)
    plt.title("")
    plt.savefig("fig/{}.png".format(file_name))
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    do_extract_LR_m1 = False
    do_extract_LR_hc = False

    n_cv = 5
    n_init = 5


    if do_extract_LR_m1:
        T_pi_vals = np.array([3])
        m1_dims = np.array([5])
        M1 = data_util.load_sabes_data('/home/fan/Data/M1/indy_20160627_01.mat')
        print("m1, input dim={}".format(M1['M1'].shape[-1]))
        xdim = 109
        good_ts = None
        dca_M1_result = run_dca_analysis(M1['M1'], M1['cursor'], T_pi_vals, dim_vals=m1_dims,
                                           num_cv_folds=n_cv, n_init=n_init, verbose=True)
        pfpc_M1_result = run_analysis_pfpc(M1['M1'], M1['cursor'], T_pi_vals, dim_vals=m1_dims, num_cv_folds=n_cv,
                                   n_init=n_init, verbose=True, Kernel=Kernel, xdim=xdim, beta=beta, beta1=beta1,
                                   beta2=beta2, good_ts=good_ts)
        with open("res/m1_vis/result_dim{}".format(5), "wb") as f:
            pickle.dump({"dca_M1_result": dca_M1_result, "pfpc_M1_result": pfpc_M1_result},f)
    else:
        with open("res/m1_vis/result_dim{}".format(5), "rb") as f:
            M1_result = pickle.load(f)
        dca_M1_result = M1_result["dca_M1_result"]
        pfpc_M1_result = M1_result["pfpc_M1_result"]

    dca_M1_test = dca_M1_result[1]['fold0']['dim5']['T_pi3']
    pfpc_M1_test = pfpc_M1_result[1]['fold0']['dim5']['T_pi3']
    dca_M1_test_ordered = dca_M1_test[:, np.argsort(dca_M1_test.std(axis=0))]
    pfpc_M1_test_ordered = pfpc_M1_test[:, np.argsort(pfpc_M1_test.std(axis=0))]
    plot_trials(dca_M1_test_ordered[:, :3], num_vis=200, file_name="dca_M1_test")
    plot_trials(pfpc_M1_test_ordered[:, :3], num_vis=200, file_name="pfpc_M1_test")


    if do_extract_LR_hc:
        T_pi_vals = np.array([5])
        hc_dims = np.array([5])
        HC = data_util.load_kording_paper_data('/home/fan/Data/HC/example_data_hc.pickle')
        print("hc, input dim={}".format(HC['neural'].shape[-1]))
        xdim = 55
        good_ts = 22000
        dca_HC_result = run_dca_analysis(HC['neural'][:good_ts], HC['loc'][:good_ts], T_pi_vals, dim_vals=hc_dims,
                                           num_cv_folds=n_cv, n_init=n_init, verbose=True)
        pfpc_HC_result = run_analysis_pfpc(HC['neural'], HC['loc'], T_pi_vals, dim_vals=m1_dims, num_cv_folds=n_cv,
                                           n_init=n_init, verbose=True, Kernel=Kernel, xdim=xdim, beta=beta, beta1=beta1,
                                           beta2=beta2, good_ts=good_ts)
        with open("res/hc_vis/result_dim{}".format(5), "wb") as f:
            pickle.dump({"dca_HC_result": dca_HC_result, "pfpc_HC_result": pfpc_HC_result},f)
    else:
        with open("res/hc_vis/result_dim{}".format(5), "rb") as f:
            HC_result = pickle.load(f)
        dca_HC_result = HC_result["dca_HC_result"]
        pfpc_HC_result = HC_result["pfpc_HC_result"]

    dca_HC_test = dca_HC_result[1]['fold0']['dim5']['T_pi5']
    pfpc_HC_test = pfpc_HC_result[1]['fold0']['dim5']['T_pi5']
    dca_HC_test_ordered = dca_HC_test[:, np.argsort(dca_HC_test.std(axis=0))]
    pfpc_HC_test_ordered = pfpc_HC_test[:, np.argsort(pfpc_HC_test.std(axis=0))]
    plot_trials(dca_HC_test_ordered[:, :3], num_vis=200, file_name="dca_HC_test")
    plot_trials(pfpc_HC_test_ordered[:, :3], num_vis=200, file_name="pfpc_HC_test")


    import pdb; pdb.set_trace()