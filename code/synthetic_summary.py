import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting.fig1 import plot_lorenz_3d, plot_lorenz_3d_colored
from synthetic_experiment import plot_latent_trials
linewidth_3d = 0.5
num_vis = 500


def plot_resps(resps, file_name, num_vis=500):
    N_to_show = 4
    label_size = 10
    ticks_size = 20
    times = np.arange(num_vis)

    fig, axs = plt.subplots(1, 1)
    # plot first stimuli plot
    y_spacing = 2  # spacing between adjacent traces on the y-axis (in data units)
    y_jump = 4  # empty y-space for elipsis
    # plot traces
    max_y = y_spacing * (N_to_show - 1) + y_jump
    offset_vals = [max_y - y_spacing * i for i in range(N_to_show)] + [0]

    n_len = resps.shape[1]
    indexes = np.random.choice(n_len, N_to_show + 1)
    for i, index in enumerate(indexes[:-1]):
        axs.plot(resps[:num_vis, index] + offset_vals[i], color='k')
    axs.plot(resps[:num_vis, index], color='k')
    axs.set_yticks(ticks=[])
    axs.set_yticks(max_y + np.array([0, -y_spacing, -2*y_spacing, -3*y_spacing, -3*y_spacing-y_jump]))
    axs.set_yticklabels(np.array([1,2,3,4,30]), fontsize=ticks_size)
    axs.set_xticks(ticks=[])
    axs.set_xlabel("", fontsize=label_size)
    axs.set_ylabel("", fontsize=label_size)
    axs.text(np.mean(times), y_jump * 0.55,
                   "ยทยทยท", rotation=90, fontsize=30, color="black",
                   horizontalalignment="center", verticalalignment="center",
                   fontweight="normal")
    plt.tight_layout()
    fig.savefig("fig/{}.png".format(file_name))
    plt.show()
    plt.close(fig)


def plot_trials(X, num_vis=200, file_name="true_lorenz_dynamics"):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plot_lorenz_3d(ax, X[:num_vis], linewidth_3d)
    plt.title("")
    plt.savefig("fig/{}.png".format(file_name))
    plt.show()
    plt.close(fig)


if __name__ == "__main__":

    do_summarization_lorenz = False
    do_summarization_lorenz_exploration = True

    if do_summarization_lorenz:
        RESULTS_FILENAME = "data/lorenz/lorenz_results.hdf5"
        # summarize R2 scores
        with open("res/lorenz_deterministic_infonce/latent_R2.pkl", "rb") as f:
            deterministic_infonce_R2s_res = pickle.load(f)
        deterministic_infonce_R2s = deterministic_infonce_R2s_res['R2_metrics']
        snr_vals = deterministic_infonce_R2s_res['snr_vals']
        with open("res/lorenz_stochastic_infonce/latent_R2.pkl", "rb") as f:
            stochastic_infonce_R2s_res = pickle.load(f)
        stochastic_infonce_R2s = stochastic_infonce_R2s_res['R2_metrics']

        with open("res/lorenz_deterministic_nwj/latent_R2.pkl", "rb") as f:
            deterministic_nwj_R2s_res = pickle.load(f)
        deterministic_nwj_R2s = deterministic_nwj_R2s_res['R2_metrics']
        with open("res/lorenz_stochastic_nwj/latent_R2.pkl", "rb") as f:
            stochastic_nwj_R2s_res = pickle.load(f)
        stochastic_nwj_R2s = stochastic_nwj_R2s_res['R2_metrics']

        with open("res/lorenz_deterministic_mine/latent_R2.pkl", "rb") as f:
            deterministic_mine_R2s_res = pickle.load(f)
        deterministic_mine_R2s = deterministic_mine_R2s_res['R2_metrics']
        with open("res/lorenz_stochastic_mine/latent_R2.pkl", "rb") as f:
            stochastic_mine_R2s_res = pickle.load(f)
        stochastic_mine_R2s = stochastic_mine_R2s_res['R2_metrics']

        with open("res/lorenz_deterministic_tuba/latent_R2.pkl", "rb") as f:
            deterministic_tuba_R2s_res = pickle.load(f)
        deterministic_tuba_R2s = deterministic_tuba_R2s_res['R2_metrics']
        with open("res/lorenz_stochastic_tuba/latent_R2.pkl", "rb") as f:
            stochastic_tuba_R2s_res = pickle.load(f)
        stochastic_tuba_R2s = stochastic_tuba_R2s_res['R2_metrics']



        # load data
        with h5py.File(RESULTS_FILENAME, "r") as f:
            snr_vals = f.attrs["snr_vals"][:]
            X = f["X"][:]
            X_dynamics = f["X_dynamics"][:]
            X_noisy_dset = f["X_noisy"][:]
            X_pca_trans_dset = f["X_pca_trans"][:]
            X_dca_trans_dset = f["X_dca_trans"][:]

        plot_resps(resps=X, file_name="30D_linear")
        plot_resps(resps=X_noisy_dset[-1], file_name="30D_linear_noisy")
        plot_trials(X=X_dynamics, num_vis=num_vis, file_name="true_lorenz_dynamics")
        # plot dca
        for i in range(4):
            plot_trials(X_dca_trans_dset[i], num_vis=num_vis, file_name="dca_snr{}".format(snr_vals[i]))
        # plot deterministic nec pfpc
        with open("res/lorenz_deterministic_infonce/inferred_trials.pkl", "rb") as f:
            deterministic_infonce_trials_res = pickle.load(f)
        deterministic_infonce_trials = deterministic_infonce_trials_res['inferred_pfpc_trials']
        for i in range(4):
            plot_trials(deterministic_infonce_trials[i], num_vis=num_vis, file_name="det_pfpc_snr{}".format(snr_vals[i]))

        # plot stochastic nec pfpc
        with open("res/lorenz_stochastic_infonce/inferred_trials.pkl", "rb") as f:
            stochastic_infonce_trials_res = pickle.load(f)
        stochastic_infonce_trials = stochastic_infonce_trials_res['inferred_pfpc_trials']
        for i in range(4):
            plot_trials(stochastic_infonce_trials[i], num_vis=num_vis, file_name="sto_pfpc_snr{}".format(snr_vals[i]))

        # plot stochastic tuba pfpc
        with open("res/lorenz_stochastic_tuba/inferred_trials.pkl", "rb") as f:
            stochastic_tuba_trials_res = pickle.load(f)
        stochastic_tuba_trials = stochastic_tuba_trials_res['inferred_pfpc_trials']
        for i in range(4):
            plot_trials(stochastic_tuba_trials[i], num_vis=num_vis, file_name="sto_pfpc_tuba_snr{}".format(snr_vals[i]))

    if do_summarization_lorenz_exploration:
        RESULTS_FILENAME = "data/lorenz/lorenz_exploration.hdf5"
        # summarize R2 scores
        with open("res/lorenz_deterministic_infonce_exploration/latent_R2.pkl", "rb") as f:
            deterministic_infonce_R2s_res = pickle.load(f)
        deterministic_infonce_exploration_R2s = deterministic_infonce_R2s_res['R2_metrics']
        snr_vals = deterministic_infonce_R2s_res['snr_vals']
        with open("res/lorenz_stochastic_infonce_exploration/latent_R2.pkl", "rb") as f:
            stochastic_infonce_R2s_res = pickle.load(f)
        stochastic_infonce_exploration_R2s = stochastic_infonce_R2s_res['R2_metrics']
        # import pdb; pdb.set_trace()
        fig = plt.figure()
        plt.plot(snr_vals, deterministic_infonce_exploration_R2s[:, 1], label="DCA")
        plt.plot(snr_vals, deterministic_infonce_exploration_R2s[:,-1], label="Deterministic")
        plt.plot(snr_vals, stochastic_infonce_exploration_R2s[:,-1], label="Stochastic")
        plt.legend()
        plt.xscale('log')
        plt.show()

        # plot latent trials
        # load data
        with h5py.File(RESULTS_FILENAME, "r") as f:
            snr_vals = f.attrs["snr_vals"][:]
            X = f["X"][:]
            X_dynamics = f["X_dynamics"][:]
            X_noisy_dset = f["X_noisy"][:]
            X_pca_trans_dset = f["X_pca_trans"][:]
            X_dca_trans_dset = f["X_dca_trans"][:]

        saved_root = "res/lorenz_deterministic_infonce_exploration"
        with open(saved_root + "/inferred_trials.pkl", "rb") as f:
            res = pickle.load(f)
        inferred_pfpc_trials = res["inferred_pfpc_trials"]
        snr_vals = res["snr_vals"]

        max_2norms = np.array([3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 1.5])
        i = -1
        for snr_val, X_pca_trans, X_dca_trans, X_noisy in zip(snr_vals, X_pca_trans_dset, X_dca_trans_dset,
                                                              X_noisy_dset):
            i += 1
            X_pfpc_trans = inferred_pfpc_trials[i]
            max_2norm = max_2norms[i]
            plot_latent_trials(X_dynamics, X_pca_trans, X_dca_trans, X_pfpc_trans, num_vis, snr_val=snr_val,
                               save_dir=saved_root, plot_lorenz_func=plot_lorenz_3d_colored, max_2norm=max_2norm)

        saved_root = "res/lorenz_stochastic_infonce_exploration"
        with open(saved_root + "/inferred_trials.pkl", "rb") as f:
            res = pickle.load(f)
        inferred_pfpc_trials = res["inferred_pfpc_trials"]
        snr_vals = res["snr_vals"]

        i = -1
        for snr_val, X_pca_trans, X_dca_trans, X_noisy in zip(snr_vals, X_pca_trans_dset, X_dca_trans_dset,
                                                              X_noisy_dset):
            i += 1
            X_pfpc_trans = inferred_pfpc_trials[i]
            max_2norm = max_2norms[i]
            plot_latent_trials(X_dynamics, X_pca_trans, X_dca_trans, X_pfpc_trans, num_vis, snr_val=snr_val,
                               save_dir=saved_root, plot_lorenz_func=plot_lorenz_3d_colored,
                               max_2norm=max_2norm)

    import pdb; pdb.set_trace()