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
import numpy as np

from helpers.sampler_wrapper import ShimmingMockSampler

class EmbeddingError(Exception):
    pass

# from helpers.embedding_helpers import raster_embedding_search
from minorminer.utils.raster_embedding import (raster_embedding_search,
                                               embeddings_to_ndarray)

def embed_loops(sampler, L, try_to_load=True, raster_breadth=2):
    """
        Attempts to expand an independent set by replacing subsets of size 'greed' with larger independent sets.
        The function iteratively tries to improve the given independent set by exploring the neighborhood of subsets.

        Parameters:
        G (networkx.Graph): The input graph.
        independent_set (list or set): A list or set of nodes forming an independent set in G.
        greed (int): The size of subsets to remove from the independent set in each iteration.

        Returns:
        list: An independent set at least as big as the original.
    """
    if not isinstance(L, int):
        raise EmbeddingError(f"'L' must be an integer. Received type {type(L).__name__}.")
    if L <= 0:
        raise EmbeddingError(f"'L' must be a positive integer. Received {L}.")

    if not isinstance(raster_breadth, int):
        raise EmbeddingError(f"'raster_breadth' must be an integer. Received type {type(raster_breadth).__name__}.")
    if raster_breadth <= 0:
        raise EmbeddingError(f"'raster_breadth' must be a positive integer. Received {raster_breadth}.")

    if not isinstance(try_to_load, bool):
        raise EmbeddingError(f"'try_to_load' must be a boolean. Received type {type(try_to_load).__name__}.")

    bqm = dimod.BinaryQuadraticModel(
        vartype='SPIN',
    )
    for spin in range(L):
        bqm.add_quadratic(spin, (spin + 1) % L, -1)
    G = dimod.to_networkx_graph(bqm)
    A = sampler.to_networkx_graph()
    
    solver_name = sampler.properties['chip_id'] # sampler.solver.name
    cache_filename = f'cached_embeddings/{solver_name}__L{L:04d}_embeddings_cached.txt'
    
    if try_to_load:
        try:
            embeddings = np.loadtxt(cache_filename, dtype=int)
            print(f'Loaded embedding from file {cache_filename}')
            return embeddings
        except (ValueError, FileNotFoundError) as e:
            print(f"Failed to load {cache_filename} with `np.loadtxt`")
            print("Error:", e)
            print("Finding embedding via raster embedding search instead.")

    # Check if the target graph has enough nodes
    if G.number_of_nodes() > A.number_of_nodes():
        raise EmbeddingError(
            f"Source graph has {G.number_of_nodes()} nodes, "
            f"which exceeds the target graph's {A.number_of_nodes()} nodes."
        )
    
    embeddings = embeddings_to_ndarray(
        raster_embedding_search(G, A, raster_breadth=raster_breadth)
        , node_order=sorted(G.nodes()))
    if embeddings.size == 0:
        raise ValueError("Embedding returned by raster_embedding_search is empty.")
    
    os.makedirs('cached_embeddings/', exist_ok=True)
    np.savetxt(cache_filename, embeddings, fmt='%d')

    return embeddings


def main():
    L = 8  # Length of chain to embed
    sampler = ShimmingMockSampler()
    try:
        embeddings = embed_loops(sampler=sampler, L=L, raster_breadth=2)
        print("Embedding successful.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
