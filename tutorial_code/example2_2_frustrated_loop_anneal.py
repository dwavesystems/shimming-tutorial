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
from tqdm import tqdm

from embed_loops import embed_loops
from helpers.helper_functions import load_experiment_data, plot_data, save_experiment_data
from helpers.paper_plotting_functions import paper_plots_example2_2


def make_fbo_dict(param, shim, embeddings):
    """Makes the FBO dict from the matrix of FBOs.

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        embeddings (List[dict]): list of embeddings

    Returns:
        dict: flux bias offsets as a dict
    """
    fbo_dict = {}
    for iemb, emb in enumerate(embeddings):
        for spin in range(param['L']):
            fbo_dict[emb[spin]] = shim['fbos'][iemb, spin]

    return fbo_dict


def make_bqm(param, shim, embeddings):
    """Makes the BQM from the matrix of coupling values.

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        embeddings (List[dict]): list of embeddings

    Returns:
        dimod.BinaryQuadraticModel: a shimmed BQM
    """

    bqm = dimod.BinaryQuadraticModel(
        vartype='SPIN',
    )
    for iemb, emb in enumerate(embeddings):
        for spin in range(param['L']):
            bqm.add_quadratic(emb[spin], emb[(spin + 1) % param['L']],
                              shim['couplings'][iemb, spin])

    return bqm


def make_logical_bqm(param, shim):
    """Makes the BQM from the matrix of coupling values.

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data

    Returns:
        dimod.BinaryQuadraticModel: a shimmed BQM
    """

    _bqm = dimod.BinaryQuadraticModel(
        vartype='SPIN',
    )
    for spin in range(param['L']):
        _bqm.add_quadratic(spin, (spin + 1) % param['L'], shim['couplings'][0, spin])

    return _bqm


def adjust_fbos(result, param, shim, embeddings, stats):
    """Adjust flux bias offsets in-place.

    Args:
        result (dimod.SampleSet): a sample set of spins used for computing statistics and adjusting shims
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        embeddings (List[dict]): list of embeddings
        stats (dict): dict of sampled statistics
    """
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


def adjust_couplings(result, param, shim, embeddings, stats):
    """Adjust couplings given a sample set.

    Args:
        result (dimod.SampleSet):  a sample set of spins used for computing statistics and adjusting shims
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        embeddings (List[dict]): list of embeddings
        stats (dict): dict of sampled statistics
    """

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
            frust_matrix[iemb, spin] = (
                (mean_correlation * np.sign(shim['nominal_couplings'][spin]) + 1) / 2
            )

    shim['couplings'] += shim['alpha_J'] * np.multiply(
        np.sign(shim['nominal_couplings']), (frust_matrix - np.mean(frust_matrix))
    )

    stats['all_couplings'].append(shim['couplings'].copy())
    stats['frust'].append(frust_matrix)


def run_iteration(param, shim, embeddings, stats):
    """Perform one iteration of the experiment, i.e., sample the BQM, adjust flux
    bias offsets and couplings, and update statistics.

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        embeddings (List[dict]): list of embeddings
        stats (dict): dict of sampled statistics
    """
    bqm = make_bqm(param, shim, embeddings)
    fbo_dict = make_fbo_dict(param, shim, embeddings)
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

    adjust_fbos(result, param, shim, embeddings, stats)
    adjust_couplings(result, param, shim, embeddings, stats)
    stats['all_alpha_Phi'].append(shim['alpha_Phi'])
    stats['all_alpha_J'].append(shim['alpha_J'])


def run_experiment(param, shim, stats, embeddings, alpha_Phi=0., alpha_J=0.):
    """Run the full experiment

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
        alpha_Phi (float, optional): learning rate for linear shims. Defaults to 0..
        alpha_J (float, optional): learning rate for coupling shims. Defaults to 0..
    """

    prefix = f'example2_2_aPhi{alpha_Phi}_aJ{alpha_J}'

    data_dict = {'param': param, 'shim': shim, 'stats': stats}
    data_dict = load_experiment_data(prefix, data_dict)

    if data_dict is not None:
        param = data_dict['param']
        shim = data_dict['shim']
        stats = data_dict['stats']

    else:
        for iteration in tqdm(range(param['num_iters']), total=param['num_iters']):
            if iteration < 100:
                shim['alpha_Phi'] = 0.
            else:
                shim['alpha_Phi'] = alpha_Phi
            if iteration < 200:
                shim['alpha_J'] = 0.
            else:
                shim['alpha_J'] = alpha_J
            run_iteration(param, shim, embeddings, stats)

        save_experiment_data(
            prefix,
            {'param': param, 'shim': shim, 'stats': stats}
        )

    plot_data(all_fbos=stats['all_fbos'], mags=stats['mags'],
              all_couplings=stats['all_couplings'], frust=stats['frust'],
              all_alpha_phi=stats['all_alpha_Phi'], all_alpha_j=stats["all_alpha_J"],
              coupler_orbits=shim['coupler_orbits'], alpha_phi=shim['alpha_Phi'], alpha_j=shim['alpha_J'],
              coupling=param["coupling"], L=param["L"])
    paper_plots_example2_2(nominal_couplings=shim['nominal_couplings'],
                           all_fbos=stats['all_fbos'],
                           all_couplings=stats['all_couplings'],
                           mags=stats['mags'],
                           frust=stats['frust'])


def main():
    """Main function to run example
    """
    param = {
        'L': 16,
        'sampler': DWaveSampler(),  # As configured
        'coupling': -0.9,  # Coupling energy scale.
        'num_iters': 300,
    }

    embeddings = embed_loops(param['L'])

    # Where the shim data (parameters and Hamiltonian terms) are stored
    shim = {
        'alpha_Phi': 0.0,
        'alpha_J': 0.0,
        'couplings': param['coupling'] * np.ones((len(embeddings), param['L']), dtype=float),
        'fbos': np.zeros((len(embeddings), param['L']), dtype=float),
        'coupler_orbits': [0] * param['L'],
    }

    # Frustrate the loops
    shim['couplings'][:, 0] *= -1

    # Save the nominal couplings so we can refer to their sign later
    shim['nominal_couplings'] = shim['couplings'][0].copy()

    # Data for plotting after the fact
    stats = {
        'mags': [],
        'frust': [],
        'all_fbos': [],
        'all_couplings': [],
        'all_alpha_Phi': [],
        'all_alpha_J': [],
    }

    run_experiment(param, shim, stats, embeddings, 0.5e-5, 5e-2)


if __name__ == "__main__":
    main()
