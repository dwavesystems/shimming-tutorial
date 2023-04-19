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

from dwave.system.samplers import DWaveSampler

from helpers.embedding_helpers import raster_embedding_search


def embed_loops(L, try_to_load=True, raster_breadth=2):
    """Embed loops of length L

    Args:
        L (int): chain length
        try_to_load (bool, optional): Flag for loading from cached data. Defaults to True.
        raster_breadth (int, optional): breadth parameter for raster embedding search. Defaults to 2.

    Returns:
        numpy.ndarray: a matrix of embeddings
    """
    sampler = DWaveSampler()  # As configured
    bqm = dimod.BinaryQuadraticModel(
        vartype='SPIN',
    )
    for spin in range(L):
        bqm.add_quadratic(spin, (spin + 1) % L, -1)
    G = dimod.to_networkx_graph(bqm)
    A = sampler.to_networkx_graph()

    cache_filename = f'cached_embeddings/{sampler.solver.name}__L{L:04d}_embeddings_cached.txt'
    if try_to_load:
        try:
            embeddings = np.loadtxt(cache_filename, dtype=int)
            print(f'Loaded embedding from file {cache_filename}')
            return embeddings
        except (ValueError, FileNotFoundError) as e:
            print(f"Failed to load {cache_filename} with `np.loadtxt`")
            print("Error:", e)
            print("Finding embedding via raster embedding search instead.")
    embeddings = raster_embedding_search(A, G, raster_breadth=raster_breadth)

    os.makedirs('cached_embeddings/', exist_ok=True)
    np.savetxt(cache_filename, embeddings, fmt='%d')

    return embeddings


def main():
    L = 8  # Length of chain to embed
    embeddings = embed_loops(L, raster_breadth=2)


if __name__ == "__main__":
    main()
