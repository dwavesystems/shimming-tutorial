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

from embed_square_lattice import embed_square_lattice
from helpers.helper_functions import (load_experiment_data, plot_data, save_experiment_data,
                                      shim_parameter_rescaling)
from helpers import orbits
from helpers.paper_plotting_functions import paper_plots_example3_2


def make_fbo_dict(embeddings, shim):
    """Makes the FBO dict from the matrix of FBOs.

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      "chain_strength" for the chain strength of embeddings,
                      "halve_boundary_couplers" a flag for whether to divide J
                      by two on the boundaries, "adaptive_step_size" flag to
                      adaptively tune step sizes for shim.
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        embeddings (List[dict]): list of embeddings

    Returns:
        dict: flux bias offsets as a dict
    """
    fbo_dict = {}
    for iemb, emb in enumerate(embeddings):
        for ispin, spin in enumerate(emb):
            fbo_dict[spin] = shim['fbos'][iemb, ispin]

    return fbo_dict


def make_bqm(shim, embeddings, logical_bqm):
    """Makes the BQM from the matrix of coupling values.

    Args:
        shim (dict): shimming data
        embeddings (List[dict]): list of embeddings
        logical_bqm (dimod.BinaryQuadraticModel): a logical BQM

    Returns:
        dimod.BinaryQuadraticModel: a shimmed BQM
    """

    _bqm = dimod.BinaryQuadraticModel(vartype='SPIN')

    for iemb, emb in enumerate(embeddings):
        for iedge, (u, v) in enumerate(logical_bqm.quadratic):
            _bqm.set_quadratic(emb[u], emb[v], shim['couplings'][iemb, iedge])

    return _bqm


def make_logical_bqm(param):
    """Makes the BQM from the matrix of coupling values.

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      "chain_strength" for the chain strength of embeddings,
                      "halve_boundary_couplers" a flag for whether to divide J
                      by two on the boundaries, "adaptive_step_size" flag to
                      adaptively tune step sizes for shim.
                      and "num_iters" for the number of shimming iterations.

    Returns:
        dimod.BinaryQuadraticModel: a shimmed BQM
    """

    _bqm = dimod.BinaryQuadraticModel(vartype='SPIN')

    for x in range(param['L']):
        for y in range(param['L']):
            _bqm.add_variable(x * param['L'] + y)

    for x in range(param['L']):
        for y in range(param['L']):
            if (x + y) % 2:
                if param['halve_boundary_couplers'] and (x == 0 or x == param['L'] - 1):
                    _bqm.set_quadratic(x * param['L'] + y,
                                       x * param['L'] + ((y + 1) % param['L']),
                                       param['coupling'] / 2)
                else:
                    _bqm.set_quadratic(x * param['L'] + y,
                                       x * param['L'] + ((y + 1) % param['L']),
                                       param['coupling'])
            else:
                _bqm.set_quadratic(x * param['L'] + y,
                                   x * param['L'] + ((y + 1) % param['L']),
                                   -param['chain_strength'] * param['coupling'])
            if x < param['L'] - 1:
                _bqm.set_quadratic(x * param['L'] + y,
                                   (x + 1) * param['L'] + y,
                                   param['coupling'])

    return _bqm


def adjust_fbos(result, param, shim, stats, embeddings):
    """Adjust flux bias offsets in-place.

    Args:
        result (dimod.SampleSet): a sample set of spins used for computing statistics and adjusting
                                  shims
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      "chain_strength" for the chain strength of embeddings,
                      "halve_boundary_couplers" a flag for whether to divide J
                      by two on the boundaries, "adaptive_step_size" flag to
                      adaptively tune step sizes for shim.
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
    """
    magnetizations = np.zeros(param['sampler'].properties['num_qubits'], dtype=float)
    used_qubit_magnetizations = result.record.sample.sum(axis=0) / len(result.record)
    for iv, v in enumerate(result.variables):
        magnetizations[v] = used_qubit_magnetizations[iv]

    shim['fbos'] -= shim['alpha_Phi'] * magnetizations[embeddings]
    stats['mags'].append(magnetizations[embeddings])
    stats['all_fbos'].append(shim['fbos'].copy())


def adjust_couplings(result, param, shim, stats, embeddings, logical_bqm):
    """Adjust couplings given a sample set.

    Args:
        result (dimod.SampleSet): a sample set of spins used for computing statistics and adjusting
                                  shims
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      "chain_strength" for the chain strength of embeddings,
                      "halve_boundary_couplers" a flag for whether to divide J
                      by two on the boundaries, "adaptive_step_size" flag to
                      adaptively tune step sizes for shim.
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
        logical_bqm (dimod.BinaryQuadraticModel): a logical BQM

    """
    vars = result.variables

    # Make a big array for the solutions, with zeros for unused qubits
    bigarr = np.zeros(shape=(param['sampler'].properties['num_qubits'], len(result)), dtype=np.int8)
    bigarr[vars, :] = dimod.as_samples(result)[0].T

    frust_matrix = np.zeros_like(shim['couplings'])

    for iemb, emb in enumerate(embeddings):
        for iedge, (u, v) in enumerate(logical_bqm.quadratic):
            mean_correlation = np.mean(np.multiply(
                bigarr[emb[u]],
                bigarr[emb[v]]
            ))
            frust_matrix[iemb, iedge] = (
                (mean_correlation * np.sign(shim['nominal_couplings'][iedge]) + 1) / 2
            )

    # Get frustration per orbit.
    _orbits = np.unique(shim['coupler_orbits'])

    for orbit in _orbits:
        # Only shim if the couplers are AFM.  Otherwise skip the orbit.
        if np.mean(shim['nominal_couplings'][shim['coupler_orbits'] == orbit]) > 0:
            mean_frust = np.mean(frust_matrix[:, shim['coupler_orbits'] == orbit])

            shim['couplings'][:, shim['coupler_orbits'] == orbit] += shim['alpha_J'] * np.multiply(
                np.sign(shim['nominal_couplings'][shim['coupler_orbits'] == orbit]),
                (frust_matrix[:, shim['coupler_orbits'] == orbit] - mean_frust)
            )

            # Renormalize to repair drift of magnitude due to truncation
            shim['couplings'][:, shim['coupler_orbits'] == orbit] *= \
                np.mean(np.divide(
                    shim['nominal_couplings'][shim['coupler_orbits'] == orbit],
                    shim['couplings'][:, shim['coupler_orbits'] == orbit]
                ))

    # Damp the couplers (push toward default value)
    if 'coupler_damp' in shim:
        excess = np.subtract(shim['couplings'],
                             shim['nominal_couplings'])
        shim['couplings'] -= excess * shim['coupler_damp']

    shim['couplings'] = np.maximum(shim['couplings'], -2.)
    shim['couplings'] = np.minimum(shim['couplings'], 1.)

    stats['all_couplings'].append(shim['couplings'].copy())
    stats['frust'].append(frust_matrix)


def get_sublattices(L):
    """Get lattice of size _L x _L

    Args:
        _L (int): linear length of lattice

    Returns:
        np.ndarray: a matrix representing 
    """
    sl = np.zeros((L * L), dtype=int)
    for x in range(L):
        for y in range(L):
            sl[x * L + y] = (np.floor((y + 3 * x) / 2)) % 3
    return sl


def compute_psi(result, param, embeddings):
    """Compute psi

    Args:
        result (dimod.SampleSet): a sample set of spins used for computing statistics and adjusting
                                  shims
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      "chain_strength" for the chain strength of embeddings,
                      "halve_boundary_couplers" a flag for whether to divide J
                      by two on the boundaries, "adaptive_step_size" flag to
                      adaptively tune step sizes for shim.
                      and "num_iters" for the number of shimming iterations.
        embeddings (List[dict]): list of embeddings

    Returns:
        np.ndarray: an array of psi values, one for each embedding
    """
    vars = result.variables

    # Make a big array for the solutions, with zeros for unused qubits
    bigarr = np.zeros(shape=(param['sampler'].properties['num_qubits'], len(result)), dtype=np.int8)
    bigarr[vars, :] = dimod.as_samples(result)[0].T

    sl = get_sublattices(param['L'])
    psi = np.zeros((len(embeddings), len(result)), dtype=complex)
    for iemb, emb in enumerate(embeddings):
        S = bigarr[emb]
        psi[iemb] = np.sqrt(3) * np.mean(np.multiply(S.T, np.exp(sl * 1j * 2 * np.pi / 3)), axis=1)

    return psi


def run_iteration(param, shim, stats, embeddings, logical_bqm):
    """Perform one iteration of the experiment, i.e., sample the BQM, adjust flux
    bias offsets and couplings, and update statistics.

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      "chain_strength" for the chain strength of embeddings,
                      "halve_boundary_couplers" a flag for whether to divide J
                      by two on the boundaries, "adaptive_step_size" flag to
                      adaptively tune step sizes for shim.
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
        logical_bqm (dimod.BinaryQuadraticModel): a logical BQM
    """
    bqm = make_bqm(shim, embeddings, logical_bqm)
    fbo_dict = make_fbo_dict(embeddings, shim)
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

    adjust_fbos(result, param, shim, stats, embeddings)
    adjust_couplings(result, param, shim, stats, embeddings, logical_bqm)
    stats['all_psi'].append(compute_psi(result, param, embeddings))
    stats['all_alpha_Phi'].append(shim['alpha_Phi'])
    stats['all_alpha_J'].append(shim['alpha_J'])


def run_experiment(param, shim, stats, embeddings, logical_bqm, alpha_Phi=0., alpha_J=0.):
    """Run the full experiment

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      "chain_strength" for the chain strength of embeddings,
                      "halve_boundary_couplers" a flag for whether to divide J
                      by two on the boundaries, "adaptive_step_size" flag to
                      adaptively tune step sizes for shim.
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
        logical_bqm (dimod.BinaryQuadraticModel): a logical BQM
        alpha_Phi (float, optional): learning rate for linear shims. Defaults to 0..
        alpha_J (float, optional): learning rate for coupling shims. Defaults to 0..
    """
    prefix = f'example3_2_{shim["type"]}{"_adaptive" * int(param["adaptive_step_size"])}' \
             f'_halve{param["halve_boundary_couplers"]}_aPhi{alpha_Phi}_aJ{alpha_J}'

    data_dict = {'param': param, 'shim': shim, 'stats': stats}
    data_dict = load_experiment_data(prefix, data_dict)

    if data_dict is not None:
        param = data_dict['param']
        shim = data_dict['shim']
        stats = data_dict['stats']

    else:
        pbar = tqdm(range(param['num_iters']), ncols=140, desc=f'PROGRESS ({prefix})')

        if not param['adaptive_step_size']:
            # Fixed step sizes

            for iteration in pbar:
                if iteration < 100:
                    shim['alpha_Phi'] = 0.
                else:
                    shim['alpha_Phi'] = alpha_Phi
                if iteration < 300:
                    shim['alpha_J'] = 0.
                else:
                    shim['alpha_J'] = alpha_J

                run_iteration(param, shim, stats, embeddings, logical_bqm)

        else:
            # Adaptive step sizes

            shim['alpha_Phi'] = alpha_Phi
            shim['alpha_J'] = alpha_J
            for iteration in pbar:
                run_iteration(param, shim, stats, embeddings, logical_bqm)
                shim['alpha_Phi'] *= shim_parameter_rescaling(stats['all_fbos'], num_iters=20,
                                                              ratio=1.1)
                shim['alpha_J'] *= shim_parameter_rescaling(stats['all_couplings'], num_iters=20,
                                                            ratio=1.1)

        save_experiment_data(
            prefix,
            {'param': param, 'shim': shim, 'stats': stats}
        )

    plot_data(all_fbos=stats['all_fbos'], mags=stats['mags'],
              all_couplings=stats['all_couplings'], frust=stats['frust'],
              all_alpha_phi=stats['all_alpha_Phi'], all_alpha_j=stats["all_alpha_J"],
              coupler_orbits=shim['coupler_orbits'], alpha_phi=shim['alpha_Phi'], alpha_j=shim['alpha_J'],
              coupling=param["coupling"], L=param["L"])
    paper_plots_example3_2(halve_boundary_couplers=param['halve_boundary_couplers'],
                           type_=shim['type'], nominal_couplings=shim['nominal_couplings'],
                           coupler_orbits=shim['coupler_orbits'], all_fbos=stats['all_fbos'],
                           all_couplings=stats['all_couplings'], mags=stats['mags'],
                           frust=stats['frust'], all_psi=stats['all_psi'])


def main():
    """Main function to run example
    """
    shimtype = 'embedded_finite'
    adaptive_step_size = False
    halve_boundary_couplers = False
    assert shimtype in ['embedded_finite', 'embedded_infinite', 'triangular_infinite']

    param = {
        'L': 12,
        'sampler': DWaveSampler(),  # As configured
        # Magnitude of coupling for FM chains, as a multiple of AFM coupling.
        'chain_strength': 2.0,
        'coupling': 0.9,  # Coupling energy scale.  Should be positive.
        'num_iters': 800,
        # Option to divide J by two on the boundaries.
        'halve_boundary_couplers': halve_boundary_couplers,
        'adaptive_step_size': adaptive_step_size,  # Option to adaptively tune step sizes for shim.
    }

    # Make the logical BQM and a bunch of disjoint embeddings
    embeddings, _ = embed_square_lattice(param['L'])

    # Make the logical BQM to get orbits for a single embedding.
    # Doing it for all embeddings together is very slow with pynauty.
    logical_bqm = make_logical_bqm(param)
    unsigned_orbits = orbits.get_orbits(logical_bqm)

    # Where the shim data (parameters and Hamiltonian terms) are stored
    shim = {
        'alpha_Phi': 0.0,
        'alpha_J': 0.0,
        'couplings': np.array([list(logical_bqm.quadratic.values())] * len(embeddings)),
        'fbos': np.zeros_like(embeddings, dtype=float),
        'type': 'embedded_infinite',
        'coupler_damp': 0.0,
    }

    # Save the nominal couplings so we can refer to their sign later
    shim['nominal_couplings'] = shim['couplings'][0].copy()

    if shim['type'] == 'embedded_finite':
        shim['coupler_orbits'] = list(unsigned_orbits[1].values())
    if shim['type'] == 'embedded_infinite':
        # Divide into three orbits: vertical FM, vertical AFM, and horizontal AFM.
        index_diff = np.array([np.abs(u - v) for (u, v) in logical_bqm.quadratic.keys()])
        shim['coupler_orbits'] = 0 * index_diff
        shim['coupler_orbits'][shim['nominal_couplings'] > 0] = 1
        shim['coupler_orbits'][index_diff == param['L']] = 2
        shim['coupler_orbits'] = list(shim['coupler_orbits'])
    if shim['type'] == 'triangular_infinite':
        shim['coupler_orbits'] = [int((1 + np.sign(x)) / 2) for x in logical_bqm.quadratic.values()]

    # Data for plotting after the fact
    stats = {
        'mags': [],
        'frust': [],
        'all_fbos': [shim['fbos'].copy()],
        'all_couplings': [shim['couplings'].copy()],
        'all_alpha_Phi': [],
        'all_alpha_J': [],
        'all_psi': [],
    }

    run_experiment(param, shim, stats, embeddings, logical_bqm, 2e-6, 0.02)


if __name__ == "__main__":
    main()
