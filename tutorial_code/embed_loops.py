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
from minorminer.utils.raster_embedding import (raster_embedding_search,
                                               embeddings_to_ndarray,
                                               raster_breadth_subgraph_lower_bound,
                                               raster_breadth_subgraph_upper_bound,
                                               subgraph_embedding_feasibility_filter)

def embed_loops(sampler: MockDWaveSampler, L: int, use_cache: bool=True, raster_breadth: int=None, 
                **re_kwargs)-> np.ndarray:
    """Embeds a ring of length L.

    Args:
        sampler (int): DWaveSampler for which to embed
        L (int): lattice length
        use_cache (bool, default=True): When True, embeddings are 
            saved to and loaded from a local directory whenever
            possible. If writing to a directory is not possible 
            a warning is thrown.

        Returns:
        np.ndarray: An matrix of embeddings
    """
    if not isinstance(L, int):
        raise ValueError(f"'L' must be an integer. Received type {type(L).__name__}.")
    if L <= 0:
        raise ValueError(f"'L' must be a positive integer. Received {L}.")
    
    if not isinstance(use_cache, bool):
        raise ValueError(f"'use_cache' must be a boolean. Received type {type(use_cache).__name__}.")

    solver_name = sampler.properties['chip_id'] # sampler.solver.name
    cache_filename = f'cached_embeddings/{solver_name}_L{L:04d}_embeddings_cached.txt'
    
    if use_cache and os.path.exists(cache_filename):
        embeddings = np.loadtxt(cache_filename, dtype=int)
        print(f'Loaded {cache_filename}')
        return embeddings

    bqm = dimod.BinaryQuadraticModel(
        vartype='SPIN',
    )
    for spin in range(L):
        bqm.add_quadratic(spin, (spin + 1) % L, -1)
    G = dimod.to_networkx_graph(bqm)
    A = sampler.to_networkx_graph()
    # Check if the target graph has enough nodes
    if not subgraph_embedding_feasibility_filter(S=G, T=A):
        raise ValueError(f'Embedding {G} on {A} is infeasible')

    if raster_breadth is None:
        raster_breadth = min(raster_breadth_subgraph_lower_bound(S=G, T=A) + 1,
                             raster_breadth_subgraph_upper_bound(T=A))
    
    if not isinstance(raster_breadth, int) or raster_breadth <= 0:
        raise ValueError(f"'raster_breadth' must be a positive integer. Received {raster_breadth}.")

    print('Creating embeddings may take several minutes.' 
          '\nTo accelerate the process a smaller lattice (L) might be '
          'considered and/or the search restricted to max_num_emb=1.')
    embeddings = embeddings_to_ndarray(
        raster_embedding_search(S=G, T=A, raster_breadth=raster_breadth,
                                max_num_emb=float('Inf'), **re_kwargs)
        , node_order=sorted(G.nodes()))

    if embeddings.size == 0:
        raise ValueError('No feasible embeddings found. '
                         '\nModifying the source (lattice) and target '
                         '(processor), or raster_embedding_search arguments '
                         'such as timeout may resolve the issue.')
    if use_cache:
        try:
            os.makedirs('cached_embeddings/', exist_ok=True)
            np.savetxt(cache_filename, embeddings, fmt='%d')
        except:
            warnings.warn('Embedding cache files could not be created')

    return embeddings


def main():
    L = 8  # Length of chain to embed
    sampler = MockDWaveSampler()
    embeddings = embed_loops(sampler=sampler, L=L, raster_breadth=2)
    if embeddings.shape[0] >= 1 and embeddings.shape[1] == L:
        print(f'{L}x{L} embedding successfully found')
    else:
        print(f'Something is wrong, {L}x{L} embedding not found')

if __name__ == "__main__":
    main()
