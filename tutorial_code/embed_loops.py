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
import dimod
import warnings
import numpy as np

from dwave.system.testing import MockDWaveSampler

from minorminer.utils.parallel_embeddings import (
    find_sublattice_embeddings,
    embeddings_to_array,
)

from minorminer.utils.feasibility import (
    embedding_feasibility_filter,
    lattice_size_lower_bound,
)


class InfeasibleResultsError(Exception):
    """Error raised when no feasible results are found."""

def embed_loops(
    sampler: MockDWaveSampler,
    L: int,
    use_cache: bool = True,
    **kwargs,
) -> np.ndarray:
    """Embeds a ring of length L.

    Args:
        sampler (int): DWaveSampler for which to embed
        L (int): Lattice length
        use_cache (bool, default=True): When True, embeddings are
            saved to and loaded from a local directory whenever
            possible. If writing to a directory is not possible
            a warning is thrown.

        Returns:
        np.ndarray: A matrix of embeddings
    """
    if not isinstance(L, int):
        raise TypeError(f"'L' must be an integer. Received type {type(L)}.")
    if L <= 0:
        raise ValueError(f"'L' must be a positive integer. Received {L}.")

    if not isinstance(use_cache, bool):
        raise TypeError(
            f"'use_cache' must be a boolean. Received type {type(use_cache)}."
        )

    solver_name = sampler.properties["chip_id"]
    cache_filename = f"cached_embeddings/{solver_name}_L{L:04d}_embeddings_cached.txt"

    if use_cache and os.path.exists(cache_filename):
        embeddings = np.loadtxt(cache_filename, dtype=int)
        print(f"Loaded {cache_filename}")
        return embeddings

    bqm = dimod.BinaryQuadraticModel(
        vartype="SPIN",
    )
    for spin in range(L):
        bqm.add_quadratic(spin, (spin + 1) % L, -1)
    G = dimod.to_networkx_graph(bqm)
    A = sampler.to_networkx_graph()

    if not embedding_feasibility_filter(S=G, T=A):
        raise ValueError(f"Embedding {G} on {A} is infeasible")

    lower_bound = lattice_size_lower_bound(S=G, T=A) + 1
    max_rows_columns = max(A.graph.get("rows"), A.graph.get("columns")
    sublattice_size = kwargs.pop("sublattice_size", min(lower_bound , max_rows_columns)))

    if not isinstance(sublattice_size, int) or sublattice_size <= 0:
        raise ValueError(
            f"'sublattice_size' must be a positive integer. Received {sublattice_size}."
        )

    print(
        "Creating embeddings may take several minutes."
        "\nTo accelerate the process a smaller lattice (L) might be "
        "considered and/or the search restricted to max_num_emb=1."
    )
    max_num_emb = kwargs.pop("max_num_emb", None)
    if max_num_emb is None:
        max_num_emb = G.number_of_nodes() // A.number_of_nodes()  # Default to many

    embedder_kwargs = {"timeout": kwargs.pop("timeout", 10)}
    embeddings = embeddings_to_array(
        find_sublattice_embeddings(
            S=G,
            T=A,
            sublattice_size=sublattice_size,
            max_num_emb=max_num_emb,
            embedder_kwargs=embedder_kwargs,
            **kwargs,
        ),
        node_order=sorted(G.nodes()),
        as_ndarray=True,
    )

    if embeddings.size == 0:
        raise InfeasibleResultsError(
            "No feasible embeddings found. "
            "\nModifying the source (lattice) and target "
            "(processor), or find_sublattice_embeddings arguments "
            "such as timeout may resolve the issue."
        )

    if use_cache:
        try:
            os.makedirs("cached_embeddings/", exist_ok=True)
            np.savetxt(cache_filename, embeddings, fmt="%d")
        except OSError as e:
            warnings.warn("Embedding cache files could not be created")

    return embeddings


def main():
    from time import perf_counter

    L = 2048  # L=2048 anticipate ~ 2.5 seconds on i7
    sampler = MockDWaveSampler(topology_type="pegasus", topology_shape=[16])
    t0 = perf_counter()
    embeddings = embed_loops(
        sampler=sampler, L=L, max_num_emb=len(sampler.nodelist)//L, use_cache=False
    )
    t1 = perf_counter() - t0
    if embeddings.size >= 1:
        print(f"Loop {L} embedding successfully found in {t1} seconds")
    else:
        print(f"Something is wrong, {L}x{L} embedding not found")


if __name__ == "__main__":
    main()
