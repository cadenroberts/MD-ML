#!/usr/bin/env python3
import pickle
import json
import os
import numpy.typing
import scipy
from report_generator.traj_loading import NativeTraj, NativeTrajPathNumpy, load_native_trajs_stride
from report_generator.tica_plots import TicaModel, calc_atom_distance
from gen_benchmark import machines
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.signal import savgol_filter
from numba import cuda
from math import exp

tica_cache_path = "/media/DATA_18_TB_1/andy/benchmark_cache/chignolin.tica"
prior_params = json.load(open("/media/DATA_18_TB_1/andy/models/benchmark_trained_trp-cage_higher_learning_rate/result-2024.11.06-18.50.45/prior_params.json", "r"))    


def cutoff(data, cutoff: int):
    n = sum(data)    
    k = 0
    i = 0
    while (k < cutoff):
        k += (data[i] / n)
        i += 1
       
    return data[:i] 


def plot(data, output_path, xlabel="index", ylabel="timescale", color="blue"):
    print(data)
    plt.scatter(range(len(data)), data, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path)
    plt.cla()


def plot_timescales(model, cutoff) -> None:
    data = model.tica_model.timescales(model.tica_model.dim)
    if cutoff is not None:
        data = cutoff(data, cutoff)

    plot(data, "./timescales.png")
    
def plot_eigenvalues(model, cutoff) -> None:
    data = model.tica_model.singular_values
    if cutoff is not None:
        data = cutoff(data, cutoff)

    plot(data, "./eigenvalues.png", ylabel="eigenvalues", color="orange")
    

def main():
    with open(tica_cache_path, 'rb') as f:
        model = pickle.load(f)
        assert isinstance(model, TicaModel)
        print("loaded tica model")
        coord_path = os.path.join(machines["bizon"].data_350_path, "chignolin_ca_coords.npy")
        native_trajs: list[NativeTraj] = load_native_trajs_stride([NativeTrajPathNumpy(coord_path, get_top_path(coord_path))], prior_params, 0, machines["bizon"].cache_path, "chignolin", False, 350)[0]
        print("loaded trajectories")
        atom_distances: list[numpy.typing.NDArray] = [calc_atom_distance(x.trajectory) for x in native_trajs]
        tica_datas: list[numpy.typing.NDArray] = model.decompose(atom_distances)
        tica_datas_all: numpy.typing.NDArray = numpy.concatenate(tica_datas)
        tica_covariances = numpy.cov(tica_datas_all.T)
        tica_max_covariance = 0
        for i in range(tica_covariances.shape[0]):
            for j in range(i):
                if i != j:
                    tica_max_covariance = max(tica_max_covariance, tica_covariances[i][j])

        """
        #Uncomment this block if you want to see covariance matrix or timescale/eigenvalue plots
        plot_timescales(model)
        plot_eigenvalues(model)
        print(f"max non-diagonal covariance is {tica_max_covariance} on matrix with shape {tica_covariances.shape}")
        print(f"covariance matrix = {tica_covariances}")
        """
        """
        # Uncomment this block if you want to generate PDF from scratch (Current params take a few minutes)
        tica_pdfs = []
        print("Calculating PDFs")
        for x in range(tica_datas_all.shape[1]):
            print(f"Calculating pdf of component {x}")
            component_tica = np.array([tica_datas_all[::50, x]])
            gpu_data = gaussian_kde_gpu(component_tica.T, np.array([tica_datas_all[::1, x]]).T)
            tica_pdfs.append(gpu_data)
        
        np.save('p_x.npy', tica_pdfs)
        """
        tica_pdfs = np.load('p_x.npy')
        print(tica_pdfs)
        k_B = 1.380649e-23  # Boltzmann constant (J/K)
        T = 350000  # Temperature in Kelvin
        energy = -k_B * T * np.sum(np.log(tica_pdfs), axis = 0) 
        print(energy)
        plot_energies(energy)

        
def plot_energies(array):
    """
    Plots a given 1D array against its indices.

    Parameters:
    - array (list or numpy array): The 1D array to plot.

    Returns:
    - None: Displays the plot.
    """
    plt.figure(figsize=(8, 5))
    indices = np.arange(len(array))
    spline = scipy.interpolate.make_interp_spline(indices, array, k=3)  # Cubic spline
    smooth_indices = np.linspace(indices[0], indices[-1], 1000)
    smooth_values = spline(smooth_indices)
    smoothed_array = savgol_filter(smooth_values, window_length=51, polyorder=3)

    
    plt.plot(smooth_indices, smoothed_array, linestyle='-', linewidth=0.5, label="Energy")
    plt.title("Energy vs Trajectory Frame")
    plt.xlabel("Trajectory Frame")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.savefig("./energy.png")
    plt.cla()

def get_top_path(coord_path: str) -> str:
    dir_path = os.path.dirname(coord_path[:-len("_coords.npy")])
    base = os.path.basename(coord_path[:-len("_coords.npy")]) + ".pdb"
    out = os.path.join(dir_path, "topology", base)
    return out

    


def bandwidth(x):
    """Scott's rule bandwidth."""
    d = x.shape[1]
    f = x.shape[0] ** (-1 / (d + 4))
    H = f * np.eye(d) * np.std(x, axis=0)
    return H*H


@cuda.jit('(float64[:], float64[:,:], float64[:,:], int64, int64, int64, float64[:], float64)')
def cuda_kernel(r, p, q, n, m, d, bw, f):
    """Numba based CUDA kernel."""
    i = cuda.grid(1, None)
    if i < m:
        for j in range(n):
            arg = 0.
            for k in range(d):
                res = p[j, k] - q[i, k]
                arg += res * res * bw[k]
            arg = f * exp(-arg / 2.)
            r[i] += arg


def gaussian_kde_gpu(p, q, threadsperblock=64):
    logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)


    """Gaussian kernel density estimation:
       Density of points p evaluated at query points q."""
    n = p.shape[0]
    d = p.shape[1]
    m = q.shape[0]
    assert d == q.shape[1]
    bw = bandwidth(p)
    bwinv = np.diag(np.linalg.inv(bw))
    bwinv = np.ascontiguousarray(bwinv)
    f = (2 * np.pi) ** (-d / 2)
    f /= np.sqrt(np.linalg.det(bw))
    d_est = cuda.to_device(np.zeros(m))
    d_p = cuda.to_device(p)
    d_q = cuda.to_device(q)
    d_bwinv = cuda.to_device(bwinv)
    blockspergrid = m // threadsperblock + 1
    cuda_kernel[blockspergrid, threadsperblock](d_est, d_p, d_q, n, m, d, d_bwinv, f)# pyright: ignore[reportIndexIssue]
    est = d_est.copy_to_host()
    est /= n
    return est

if __name__ == "__main__":
    main()
