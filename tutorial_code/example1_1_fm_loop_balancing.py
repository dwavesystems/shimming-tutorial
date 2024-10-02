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

# working with binary quadratic models, essential for formulating Ising problems
import dimod
# for numerical operations 
import numpy as np

from helpers.sampler_wrapper import ShimmingMockSampler
from dwave.system.samplers import DWaveSampler
# display progress bar during iterations 
from tqdm import tqdm

# embedding the FM loop into the QPUs topology
from embed_loops import embed_loops
# functions to load, save, and plot experiment data
from helpers.helper_functions import load_experiment_data, plot_data, save_experiment_data
from helpers.paper_plotting_functions import paper_plots_example1_1

# constructs a dictionary mapping each qubit in the QPU to its corresponding flux bias offset value
# iterates over each embedding and each spin withiin the loop
# assign the corresponding FBO from the shim data to the specific qubit in QPU
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

# costructs a bqm representing the fm loop with updated couplings
# prepares he bqm for the QPU to solve, incorporating the current shimming state to ensure 
# uniform frustration probabilities across couplers
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

# adjust the flux bias offsets to minimize the average magnetization of each qubit. 
# Iterative gradient descen method described in the paper for balancing qubit magnetizations, 
# ensuring that each qubit's average magnetization approaches zero
def adjust_fbos(result, param, shim, stats, embeddings):
    """Adjust flux bias offsets in-place.

    Args:
        result (dimod.SampleSet): a sample set of spins used for computing statistics and adjusting shims
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
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

# adjust the coupler strengths to ensure uniform frustration probabilities across all couplers 
def adjust_couplings(result, param, shim, stats, embeddings):
    """Adjust couplings given a sample set.

    Args:
        result (dimod.SampleSet):  a sample set of spins used for computing statistics and adjusting shims
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
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
            frust_matrix[iemb, spin] = (mean_correlation * np.sign(param['coupling']) + 1) / 2

    shim['couplings'] += (np.sign(param['coupling'])
                          * shim['alpha_J']
                          * (frust_matrix - np.mean(frust_matrix)))
    stats['all_couplings'].append(shim['couplings'].copy())
    stats['frust'].append(frust_matrix)

# encapsulates a single iteration of the calibration process, performing sampling, adjusting fbos and couplings
def run_iteration(param, shim, stats, embeddings):
    """Perform one iteration of the experiment, i.e., sample the BQM, adjust flux
    bias offsets and couplings, and update statistics.

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
    """
    bqm = make_bqm(param, shim, embeddings)
    fbo_dict = make_fbo_dict(param, shim, embeddings)

    # if 'fbos' in shim:
    #   flux_biases = shim['fbos'] 
    # else:
    flux_biases = [0] * param['sampler'].properties['num_qubits']

    for qubit, fbo in fbo_dict.items():
        flux_biases[qubit] = fbo
    result = param['sampler'].sample(
        bqm,
        annealing_time=1.0,
        num_reads=100,
        readout_thermalization=100.,
        auto_scale=False,
        flux_drift_compensation=True,
        flux_biases=flux_biases,
        answer_mode="raw",
    )
    adjust_fbos(result, param, shim, stats, embeddings)
    adjust_couplings(result, param, shim, stats, embeddings)
    stats['all_alpha_Phi'].append(shim['alpha_Phi'])
    stats['all_alpha_J'].append(shim['alpha_J'])

# executes the entire calibration experiment, managing data loading/saving and plotting 
def run_experiment(param, shim, stats, embeddings, _alpha_Phi=0., _alpha_J=0.):
    """Run the full experiment

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
        _alpha_Phi (float, optional): learning rate for linear shims. Defaults to 0..
        _alpha_J (float, optional): learning rate for coupling shims. Defaults to 0..
    """
    prefix = f'example1_1_aPhi{_alpha_Phi}_aJ{_alpha_J}'

    data_dict = {'param': param, 'shim': shim, 'stats': stats}
    data_dict = load_experiment_data(prefix, data_dict)
    if data_dict is not None:
        param = data_dict['param']
        shim = data_dict['shim']
        stats = data_dict['stats']

    else:
        print("Running experiment")
        for iteration in tqdm(range(param['num_iters']), total=param['num_iters']):
            if iteration < param['num_iters_unshimmed']:
               shim['alpha_Phi'] = 0.
            else:
               shim['alpha_Phi'] = _alpha_Phi
            """
            if iteration < 10:
                shim['alpha_Phi'] = _alpha_Phi * 2
            else:
                shim['alpha_Phi'] = _alpha_Phi
            """
            shim['alpha_J'] = _alpha_J
            run_iteration(param, shim, stats, embeddings)

        save_experiment_data(
            prefix,
            {'param': param, 'shim': shim, 'stats': stats}
        )
       

    plot_data(all_fbos=stats['all_fbos'], mags=stats['mags'],
              all_couplings=stats['all_couplings'], frust=stats['frust'],
              all_alpha_phi=stats['all_alpha_Phi'], all_alpha_j=stats["all_alpha_J"],
              coupler_orbits=shim['coupler_orbits'], alpha_phi=shim['alpha_Phi'], alpha_j=shim['alpha_J'],
              coupling=param["coupling"], L=param["L"])
    paper_plots_example1_1(alpha_phi=shim['alpha_Phi'],
                           all_fbos=stats['all_fbos'], mags=stats['mags'])


def main(sampler_type='mock', model_type='independent_spins'):
    """Main function to run example

    Args:
        sampler_type (string, optional): option to specify sampler type. Defaults to MockDWaveSampler.
    """

    if sampler_type == 'mock':
        sampler_instance = ShimmingMockSampler()
        sampler = sampler_instance.get_sampler()
    else:
        sampler = DWaveSampler()
    
    # Determine the number of qubits in the QPU
    num_programmed_variables = len(sampler.nodelist)
    
    # Each qubit is treated as an independent unit.  Embedding is a list of list,
    # where each iner list contains a single qubit from the nodelist. 
    if model_type == 'independent_spins':
        embeddings = [[n for n in sampler.nodelist]]
        coupling = 0  
    else:
        coupling = -0.2
        embeddings = embed_loops(param['L'], sampler.to_networkx_graph())  

    num_iters_unshimmed = 10
    num_iters = 20
    for alpha_Phi in [1e-4, 1e-5, 1e-6]:
        param = {
            'L':16,
            'sampler': sampler,  # As configured
            'coupling': coupling,  # Coupling energy scale.
            'num_iters': num_iters,
            'num_iters_unshimmed': num_iters_unshimmed,
        }

        embeddings = embed_loops(param['L'], sampler = param['sampler'])

        # Where the shim data (parameters and Hamiltonian terms) are stored
        shim = {
            'alpha_Phi': 0.0,
            'alpha_J': 0.0,
            'couplings': param['coupling'] * np.ones((len(embeddings), param['L']), dtype=float),
            'fbos': -100e-6 * np.ones((len(embeddings), param['L']), dtype=float),  # offset here, then it should return to 0
            # 'fbos': np.zeros((len(embeddings), param['L']), dtype=float),
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

        run_experiment(param, shim, stats, embeddings, alpha_Phi, 0.)

if __name__ == "__main__":
    main()
