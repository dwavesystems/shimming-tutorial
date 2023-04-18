# Copyright 2023 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import copy

from pathlib import Path

import numpy as np
import pickle
import lzma
import matplotlib

from matplotlib import pyplot as plt


def movmean(x, w):
    """Takes a moving mean along the first axis, padded at the front.

    Args:
        x (np.ndarray): the input array to compute a moving mean for
        w (int): moving mean window size

    Returns:
        np.ndarray: an array of the same shape as `x` with rolling means computed along the first axis
    """

    D = np.reshape(x, (x.shape[0], -1))
    Q = np.tile(np.atleast_2d(D[1, :]).T, (1, w)).T

    ret = np.zeros_like(D)
    for i in range(len(D)):
        ret[i, :] = np.mean(D[np.maximum(0, np.arange(i - w + 1, i + 1)), :], axis=0)
    ret = np.reshape(ret, x.shape)

    return ret


def shim_parameter_rescaling(statistic, num_iters=20, ratio=1.1, tol=0.1):
    """Returns a rescaling for a shim parameter based on a lookback of num_iters iterations
    based on statistic, which is presumed to be either a list or ndarray whose first dimension
    represents iterations.

    :param statistic: Recorded shim history of the parameters being shimmed
    :param num_iters: Number of iterations to look back over to calculate the exponent
    :param ratio: Multiplier for increasing step size ( *= 1/ratio if decreasing)
    :param tol: Tolerance about 0.5, within which step size is unchanged
    :return: scaling factor
    """

    if len(statistic) >= num_iters:

        M = np.array(statistic)
        X = np.flip(np.var(M - M[-1], axis=(1, 2)))
        exponent = np.polyfit(np.log(np.arange(1, num_iters)), np.log(X[1:num_iters]), 1)[0]

        if exponent > 1.0 + tol:
            return ratio
        if exponent < 1.0 - tol:
            return 1 / ratio

    return 1.0


def plot_data(param, shim, stats,):
    """Plots data

    Args:
        param (dict): parameters of data
        shim (dict): shimmed values
        stats (dict): statistics
    """
    plt.rcParams['figure.figsize'] = (18, 10)
    fig = plt.figure(1)
    plt.clf()
    axs = fig.subplots(2, 6)

    plt.sca(axs[0, 0])
    plt.plot(
        np.reshape(np.array(stats['all_fbos']), (len(stats['all_fbos']), -1))[:, ::10]
    )
    plt.title('FBOs')

    M = np.array(stats['mags'])
    Y = movmean(M, 10)
    plt.sca(axs[0, 1])
    plt.plot(
        np.std(np.reshape(Y[10:], (len(Y[10:]), -1)), axis=1)
    )
    plt.title('std of 10-call moving mean of m')

    plt.sca(axs[0, 2])
    plt.hist(Y[10].ravel(), alpha=0.5)
    plt.hist(Y[-1].ravel(), alpha=0.5)

    plt.sca(axs[0, 3])
    plt.plot(
        # Plots the mean abs difference in magnetization from one call to the next...
        np.mean(np.abs(np.diff(M, axis=0)), axis=(1, 2))
    )
    plt.title('mean abs difference in m')

    plt.sca(axs[1, 0])
    plt.plot(
        np.reshape(
            np.divide(np.array(stats['all_couplings']), stats['all_couplings'][0]),
            (len(stats['all_couplings']), -1))[:, ::10]
    )
    plt.title('couplings (vs. nominal)')

    M = np.array(stats['frust'])
    Y = movmean(M, 10)

    # Get Y for orbits
    orbits = np.unique(shim['coupler_orbits'])
    Y_orbit = np.zeros((Y.shape[0], len(orbits)))

    for iorbit, orbit in enumerate(orbits):
        mymat = Y[:, :, shim['coupler_orbits'] == orbit]
        Y_orbit[:, iorbit] = np.std(mymat, axis=(1, 2))

    plt.sca(axs[1, 1])

    plt.plot(
        np.mean(Y_orbit[10:], axis=1)
    )
    plt.title('std of 10-call moving mean of f; avg per orbit')

    plt.sca(axs[1, 2])
    plt.hist(Y[10].ravel(), alpha=0.5)
    plt.hist(Y[-1].ravel(), alpha=0.5)

    plt.sca(axs[1, 3])
    plt.plot(
        # Plots the mean abs difference in magnetization from one call to the next...
        np.mean(np.abs(np.diff(M, axis=0)), axis=(1, 2))
    )
    plt.title('mean abs difference in f')

    # Convergence of FBOs
    M = np.array(stats['all_fbos'])

    # Plot mean sign diff
    plt.sca(axs[0, 4])
    plist = []
    for t in np.flip(np.arange(len(M) - 19)):
        X = np.flip(np.var(M[-1 - t - 19:-1 - t] - M[-1 - t], axis=(1, 2)))
        if min(np.abs(X)) == 0:
            plist.append(0.)
        else:
            plist.append(np.polyfit(np.log(np.arange(1, 20)), np.log(X), 1)[0])
    plt.plot(np.arange(19, len(M)), plist)
    plt.title('fluctuation variance exponent')

    plt.sca(axs[0, 5])
    plt.plot(np.array(stats['all_alpha_Phi']))
    plt.yscale('log')
    plt.title('$\\alpha_\Phi$')

    # Convergence of couplings
    M = np.array(stats['all_couplings'])

    # Plot 20-iteration trailing fluctuation variance exponent
    plt.sca(axs[1, 4])
    plist = []
    for t in np.flip(np.arange(len(M) - 19)):
        X = np.flip(np.var(M[-1 - t - 19:-1 - t] - M[-1 - t], axis=(1, 2)))
        if min(np.abs(X)) == 0:
            plist.append(0.)
        else:
            plist.append(np.polyfit(np.log(np.arange(1, 20)), np.log(X), 1)[0])
    plt.plot(np.arange(19, len(M)), plist)
    plt.title('fluctuation variance exponent')

    plt.sca(axs[1, 5])
    plt.plot(np.array(stats['all_alpha_J']))
    plt.yscale('log')
    plt.title('$\\alpha_J$')

    plt.suptitle(f'J={param["coupling"]}, L={param["L"]}, alpha_Phi={shim["alpha_Phi"]}, alpha_J={shim["alpha_J"]}')
    plt.tight_layout()
    plt.show()


def get_coupler_colors(_G, _bqm):
    """Uses normalized weights in the BQM to create a list of colour maps

    Args:
        _G (nx.Graph): a graph
        _bqm (dimod.BQM): a bqm whose weights are used to determine colours of the graph

    Returns:
        list[tuple[float]]: list of tuples whose coordinates represent colours used in matplotlib
    """
    cm = matplotlib.cm.get_cmap(name='coolwarm')
    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    return [cm(norm(_bqm.quadratic[E])) for E in _G.edges()]


def get_qubit_colors(_G, _bqm):
    """Uses normalized weights in the BQM to create a list of colour maps

    Args:
        _G (nx.Graph): a graph
        _bqm (dimod.BQM): a bqm whose weights are used to determine colours of the graph

    Returns:
        list[tuple[float]]: list of tuples whose coordinates represent colours used in matplotlib
    """
    cm = matplotlib.cm.get_cmap(name='coolwarm')
    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    return [cm(norm(_bqm.linear[V])) for V in _G.nodes()]


def load_experiment_data(prefix, data_dict):
    """Load a dictionary of data frome filepath.

    Args:
        prefix (str): prefix used in naming cached data.
        data_dict (dict): dictionary to populate with experiment data.

    Returns:
        dict: dictionary of experiment data.
    """
    filename = 'savedata_' + prefix + '.pkl'
    filepath = Path('cached_experiment_data').joinpath(''.join(filename))

    if not os.path.exists(filepath):
        print(f'{filepath} not found.  Couldn''t load data.')
        return None

    with lzma.open(filepath, 'rb') as f:
        loaded_data_dict = pickle.load(f)

    temp = None
    if 'param' in data_dict and 'sampler' in data_dict['param']:
        temp = data_dict['param']['sampler']
    for key in data_dict:
        data_dict[key] = loaded_data_dict[key]
        if temp is not None:
            data_dict['param']['sampler'] = temp

    print(f'Loaded {filepath}')
    return data_dict


def save_experiment_data(prefix, data_dict, overwrite=True):
    """Save experiment data.

    Args:
        prefix (str): prefix used in naming cached data.
        data_dict (dict): dictionary of data to store.
        overwrite (bool, optional): Flag for overwriting stored data. Defaults to True.

    Returns:
        bool: indicator for successful storage of data.
    """
    filename = 'savedata_' + prefix + '.pkl'
    filepath = Path('cached_experiment_data').joinpath(''.join(filename))

    if os.path.exists(filepath) and not overwrite:
        print(f'{filepath} exists.  Not overwriting.')
        return False

    for key in data_dict:
        data_dict[key] = copy.copy(data_dict[key])

    # Need to remove some sampler fields to make the data serializable.
    if 'param' in data_dict and 'sampler' in data_dict['param']:
        data_dict['param']['sampler'].solver.client = None
        data_dict['param']['sampler'].client = None

    os.makedirs('cached_experiment_data', exist_ok=True)
    with lzma.open(filepath, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f'Saved {filepath}')
    return True
