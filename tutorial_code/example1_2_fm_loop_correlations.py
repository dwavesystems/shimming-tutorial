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
import dimod
import numpy as np

from dwave.system.samplers import DWaveSampler

from embed_loops import embed_loops

from helpers.helper_functions import plot_data, save_experiment_data, load_experiment_data
from helpers.paper_plotting_functions import paper_plots_example1_2

from tqdm import tqdm


def make_fbo_dict():
    """Makes the FBO dict from the matrix of FBOs."""
    fbo_dict = {}
    for iemb, emb in enumerate(embeddings):
        for spin in range(param['L']):
            fbo_dict[emb[spin]] = shim['fbos'][iemb, spin]

    return fbo_dict


def make_bqm():
    """Makes the BQM from the matrix of coupling values."""

    bqm = dimod.BinaryQuadraticModel(
        vartype='SPIN',
    )
    for iemb, emb in enumerate(embeddings):
        for spin in range(param['L']):
            bqm.add_quadratic(emb[spin], emb[(spin + 1) % param['L']], shim['couplings'][iemb, spin])

    return bqm


def adjust_fbos(result):
    magnetizations = [0] * param['sampler'].properties['num_qubits']
    used_qubit_magnetizations = result.record.sample.sum(axis=0) / len(result.record)
    for iv, v in enumerate(result.variables):
        magnetizations[v] = used_qubit_magnetizations[iv]

    mag_array = np.zeros_like(shim['fbos'])
    for iemb in range(len(embeddings)):
        for iqubit in range(param['L']):
            mag_array[iemb, iqubit] = magnetizations[embeddings[iemb][iqubit]]

    shim['fbos'] -= shim['alpha_Phi'] * mag_array

    stats['mags'].append(mag_array)
    stats['all_fbos'].append(shim['fbos'].copy())


def adjust_couplings(result):
    vars = result.variables

    # Make a big array for the solutions, with zeros for unused qubits
    bigarr = np.zeros(shape=(param['sampler'].properties['num_qubits'], len(result)), dtype=np.int8)
    bigarr[vars, :] = dimod.as_samples(result)[0].T

    frust_matrix = np.zeros_like(shim['couplings'])

    for iemb, emb in enumerate(embeddings):
        for spin in range(param['L']):
            mean_correlation = np.mean(np.multiply(
                bigarr[emb[spin]],
                bigarr[emb[(spin + 1) % param['L']]]
            ))
            frust_matrix[iemb, spin] = (mean_correlation * np.sign(param['coupling']) + 1) / 2

    shim['couplings'] += np.sign(param['coupling']) * shim['alpha_J'] * (frust_matrix - np.mean(frust_matrix))

    stats['all_couplings'].append(shim['couplings'].copy())
    stats['frust'].append(frust_matrix)


def run_iteration():
    bqm = make_bqm()
    fbo_dict = make_fbo_dict()
    fbo_list = [0] * param['sampler'].properties['num_qubits']
    for qubit, fbo in fbo_dict.items():
        fbo_list[qubit] = fbo

    result = param['sampler'].sample(
        bqm,
        annealing_time=1.0,
        num_reads=100,
        readout_thermalization=100.,
        auto_scale=False,
        flux_drift_compensation=True,
        flux_biases=fbo_list,
        answer_mode="raw",
    )

    adjust_fbos(result)
    adjust_couplings(result)
    stats['all_alpha_Phi'].append(shim['alpha_Phi'])
    stats['all_alpha_J'].append(shim['alpha_J'])


def run_experiment(alpha_Phi=0., alpha_J=0.):
    global param
    global shim
    global stats

    prefix = f'example1_2_aPhi{alpha_Phi}_aJ{alpha_J}'

    data_dict = {'param': param, 'shim': shim, 'stats': stats}
    data_dict = load_experiment_data(prefix, data_dict)

    if data_dict is not None:
        param = data_dict['param']
        shim = data_dict['shim']
        stats = data_dict['stats']

    else:
        for iter in tqdm(range(param['num_iters']), total=param['num_iters']):
            if iter < 100:
                shim['alpha_Phi'] = 0.
            else:
                shim['alpha_Phi'] = alpha_Phi
            if iter < 200:
                shim['alpha_J'] = 0.
            else:
                shim['alpha_J'] = alpha_J
            run_iteration()

        save_experiment_data(
            prefix,
            {'param': param, 'shim': shim, 'stats': stats}
        )

    plot_data(param, shim, stats)
    paper_plots_example1_2(param, shim, stats)


if __name__ == "__main__":
    global param
    global shim
    global stats

    param = {
        'L': 64,
        'sampler': DWaveSampler(),  # As configured
        'coupling': -0.2,  # Coupling energy scale.
        'num_iters': 300,
    }

    embeddings = embed_loops(param['L'])

    # Where the shim data (parameters and Hamiltonian terms) are stored
    shim = {
        'alpha_Phi': 0.0,
        'alpha_J': 0.0,
        'couplings': param['coupling'] * np.ones((len(embeddings), param['L']), dtype=float),
        'fbos': np.zeros((len(embeddings), param['L']), dtype=float),
        'coupler_orbits': [0] * param['L'],  # We manually set all couplers to the same orbit.
    }

    # Data for plotting after the fact
    stats = {
        'mags': [],
        'frust': [],
        'all_fbos': [],
        'all_couplings': [],
        'all_alpha_Phi': [],
        'all_alpha_J': [],
    }

    run_experiment(1e-5, 5e-3)
