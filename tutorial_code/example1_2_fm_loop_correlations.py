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

from typing import Optional

from tqdm import tqdm
import numpy as np

import dimod
from dwave.system.samplers import DWaveSampler

from embed_loops import embed_loops
from helpers.helper_functions import load_experiment_data, save_experiment_data
from helpers.paper_plotting_functions import paper_plots_example1_2
from helpers.sampler_wrapper import ShimmingMockSampler


def make_fbo_dict(param: dict, shim: dict, embeddings: list) -> dict:
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
    fbo_dict = {
        emb[spin]: shim["fbos"][iemb, spin]
        for iemb, emb in enumerate(embeddings)
        for spin in range(param["L"])
    }

    return fbo_dict


def make_bqm(param: dict, shim: dict, embeddings: list) -> dimod.BinaryQuadraticModel:
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
        vartype="SPIN",
    )
    for iemb, emb in enumerate(embeddings):
        for spin in range(param["L"]):
            bqm.add_quadratic(
                emb[spin], emb[(spin + 1) % param["L"]], shim["couplings"][iemb, spin]
            )

    return bqm


def adjust_fbos(
    result: dimod.SampleSet, param: dict, shim: dict, stats: dict, embeddings: list
) -> None:
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
    magnetizations = [0] * param["sampler"].properties["num_qubits"]
    used_qubit_magnetizations = result.record.sample.sum(axis=0) / len(result.record)
    for iv, v in enumerate(result.variables):
        magnetizations[v] = used_qubit_magnetizations[iv]

    mag_array = np.zeros_like(shim["fbos"])
    for iemb in range(len(embeddings)):
        for iqubit in range(param["L"]):
            mag_array[iemb, iqubit] = magnetizations[embeddings[iemb][iqubit]]

    shim["fbos"] -= shim["alpha_Phi"] * mag_array

    stats["mags"].append(mag_array)
    stats["all_fbos"].append(shim["fbos"].copy())


def adjust_couplings(
    result: dimod.SampleSet, param: dict, shim: dict, stats: dict, embeddings: list
) -> None:
    """Adjust couplings given a sample set.

    Args:
        result (dimod.SampleSet): a sample set of spins used for computing statistics and adjusting shims
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
    """
    vars = result.variables

    # Make an array large enough for the solutions, with zeros for unused qubits
    bigarr = np.zeros(
        shape=(param["sampler"].properties["num_qubits"], len(result)), dtype=np.int8
    )
    bigarr[vars, :] = dimod.as_samples(result)[0].T

    frust_matrix = np.zeros_like(shim["couplings"])

    for iemb, emb in enumerate(embeddings):
        for spin in range(param["L"]):
            mean_correlation = np.mean(
                np.multiply(bigarr[emb[spin]], bigarr[emb[(spin + 1) % param["L"]]])
            )
            frust_matrix[iemb, spin] = (
                mean_correlation * np.sign(param["coupling"]) + 1
            ) / 2

    shim["couplings"] += (
        np.sign(param["coupling"])
        * shim["alpha_J"]
        * (frust_matrix - np.mean(frust_matrix))
    )
    stats["all_couplings"].append(shim["couplings"].copy())
    stats["frust"].append(frust_matrix)


def run_iteration(param: dict, shim: dict, stats: dict, embeddings: list) -> None:
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
    flux_biases = [0] * param["sampler"].properties["num_qubits"]

    for qubit, fbo in fbo_dict.items():
        flux_biases[qubit] = fbo

    result = param["sampler"].sample(
        bqm,
        annealing_time=1.0,
        num_reads=100,
        readout_thermalization=100.0,
        auto_scale=False,
        flux_drift_compensation=True,
        flux_biases=flux_biases,
        answer_mode="raw",
    )

    adjust_fbos(result, param, shim, stats, embeddings)
    adjust_couplings(result, param, shim, stats, embeddings)
    stats["all_alpha_Phi"].append(shim["alpha_Phi"])
    stats["all_alpha_J"].append(shim["alpha_J"])


def run_experiment(
    param: dict,
    shim: dict,
    stats: dict,
    embeddings: list,
    alpha_Phi: float = 0.0,
    alpha_J: float = 0.0,
    use_cache: bool = True,
) -> dict:
    """Run the full experiment

    Args:
        param (dict): parameters with keys "L" for length, "sampler" for
                      sampler (QPU), "coupling" for the coupling energy scale,
                      and "num_iters" for the number of shimming iterations.
        shim (dict): shimming data
        stats (dict): dict of sampled statistics
        embeddings (List[dict]): list of embeddings
        alpha_Phi (float): learning rate for linear shims. Defaults to 0.
        alpha_J (float): learning rate for coupling shims. Defaults to 0.
        use_cache (bool): When True an attempt is made to load (save) data from
            (to) the directory cached_experimental_data.
    """
    if use_cache:
        solver_name = param["sampler"].properties["chip_id"]
        prefix = f"{solver_name}_example1_2_aPhi{alpha_Phi}_aJ{alpha_J}"
        data_dict = {"param": param, "shim": shim, "stats": stats}
        data_dict = load_experiment_data(prefix, data_dict)
    else:
        data_dict = None
    if data_dict is not None:
        shim = data_dict["shim"]
        stats = data_dict["stats"]
    else:
        # prev_execution_time = 182.0092 sec.
        print("Collection of data typically requires several minutes")
        for iteration in tqdm(range(param["num_iters"]), total=param["num_iters"]):
            if iteration < param["num_iters_unshimmed_flux"]:
                shim["alpha_Phi"] = 0.0
            else:
                shim["alpha_Phi"] = alpha_Phi
            if iteration < param["num_iters_unshimmed_J"]:
                shim["alpha_J"] = 0.0
            else:
                shim["alpha_J"] = alpha_J
            run_iteration(param, shim, stats, embeddings)
        if use_cache:
            save_experiment_data(prefix, {"shim": shim, "stats": stats})

    paper_plots_example1_2(
        all_couplings=stats["all_couplings"],
        all_fbos=stats["all_fbos"],
        mags=stats["mags"],
        frust=stats["frust"],
    )


def main(
    solver_name: str = None,
    coupling: float = -0.2,
    num_iters: int = 300,
    num_iters_unshimmed_flux: int = 100,
    num_iters_unshimmed_J: int = 200,
    max_num_emb: Optional[int] = None,
    L=16,
    use_cache: bool = True,
) -> None:
    """Main function to run example.

    Completes an experiment matched to Figure 7 of DOI:10.3389/fcomp.2023.1238988,
    plotting a corresponding figure.

    Args:
        solver_name (string): option to specify sampler type. The default client QPU
            is used by default other options are listed in Leap, to use a locally executed
            classical placeholder for debugging select 'MockDWaveSampler'.
        coupling (float): coupling strength on chain.
        num_iters (int): Number of sequential programmings.
        num_iters_unshimmed_flux (int): Number of sequential programmings without flux shimming.
        num_iters_unshimmed_J (int): Number of sequential programmings without J shimming.
        max_num_emb (optional, int): Maximum number of embeddings to use per
            programming. Published tutorial data uses the maximum number the
            process can accommodate.
        use_cache (bool): When True embeddings and data are read from
            (and saved to) local directories, repeated executions can reuse
            collected data. When False embeddings and data are recalculated on
            each call. Defaults to True
    """
    if solver_name == "MockDWaveSampler":
        sampler = ShimmingMockSampler()
    else:
        sampler = DWaveSampler(solver=solver_name)

    if max_num_emb is None:
        max_num_emb = len(sampler.nodelist)//L

    param = {
        "L": L,
        "sampler": sampler,
        "coupling": coupling,
        "num_iters": num_iters,
        "num_iters_unshimmed_flux": num_iters_unshimmed_flux,
        "num_iters_unshimmed_J": num_iters_unshimmed_J,
    }

    embeddings = embed_loops(
        sampler=sampler, L=param["L"], max_num_emb=max_num_emb, use_cache=True
    )

    # Where the shim data (parameters and Hamiltonian terms) are stored
    shim = {
        "alpha_Phi": 0.0,
        "alpha_J": 0.0,
        "couplings": param["coupling"]
        * np.ones((len(embeddings), param["L"]), dtype=float),
        "fbos": np.zeros((len(embeddings), param["L"]), dtype=float),
        "coupler_orbits": [0] * param["L"],
    }

    # Data for plotting after the fact
    stats = {
        "mags": [],
        "frust": [],
        "all_fbos": [],
        "all_couplings": [],
        "all_alpha_Phi": [],
        "all_alpha_J": [],
    }
    run_experiment(param, shim, stats, embeddings, 1e-5, 5e-3, use_cache)


if __name__ == "__main__":
    main()
