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
from dwave.system.samplers import DWaveSampler
import numpy as np
from helpers.embedding_helpers import raster_embedding_search
import os


def embed_loops(_L, try_to_load=True, raster_breadth=2):
    sampler = DWaveSampler()  # As configured
    bqm = dimod.BinaryQuadraticModel(
        vartype='SPIN',
    )
    for spin in range(_L):
        bqm.add_quadratic(spin, (spin + 1) % _L, -1)
    G = dimod.to_networkx_graph(bqm)
    A = sampler.to_networkx_graph()

    cache_filename = f'cached_embeddings/{sampler.solver.name}__L{_L:04d}_embeddings_cached.txt'
    if try_to_load:
        try:
            embeddings = np.loadtxt(cache_filename, dtype=int)
            print(f'Loaded embedding from file {cache_filename}')
            return embeddings
        except:
            pass

    embeddings = raster_embedding_search(A, G, raster_breadth=raster_breadth)

    os.makedirs('cached_embeddings/', exist_ok=True)
    np.savetxt(cache_filename, embeddings, fmt='%d')

    return embeddings


if __name__ == "__main__":
    L = 8  # Length of chain to embed
    embeddings = embed_loops(L, raster_breadth=2)
    pass
