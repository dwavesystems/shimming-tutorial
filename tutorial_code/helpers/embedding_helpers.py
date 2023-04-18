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
import time

import dimod
import dwave_networkx as dnx
import networkx as nx
import numpy as np

from dwave import embedding
from dwave.system.samplers import DWaveSampler
from minorminer import subgraph as glasgow


def get_pegasus_subgrid(A, rows, cols, gridsize=16):
    """Make a subgraph of a Pegasus-16 (Advantage) graph on a set of rows and columns of unit cells.

    Args:
        A (nx.Graph): Qubit connectivity graph
        rows (Iterable): Iterable of rows of unit cells to include
        cols (Iterable): Iterable of columns of unit cells to include
    Returns:
        nx.Graph: The subgraph of A induced on the nodes in "rows" and "cols"
    """

    coords = [dnx.pegasus_coordinates(gridsize).linear_to_nice(v) for v in A.nodes]
    used_coords = [c for c in coords if c[1] in cols and c[2] in rows]

    return A.subgraph([dnx.pegasus_coordinates(gridsize).nice_to_linear(c) for c in used_coords]).copy()


def get_zephyr_subgrid(A, rows, cols, gridsize=4):
    """Make a subgraph of a Zephyr (Advantage2) graph on a set of rows and columns of unit cells.

    Args:
        A (nx.Graph): Qubit connectivity graph
        rows (Iterable): Iterable of rows of unit cells to include
        cols (Iterable): Iterable of columns of unit cells to include

    Returns:
        nx.Graph: The subgraph of A induced on the nodes in "rows" and "cols"
    """

    coords = [dnx.zephyr_coordinates(gridsize).linear_to_zephyr(v) for v in A.nodes]
    c = np.asarray(coords)
    used_coords = [c for c in coords if
                   (c[0] == 0 and c[4] in cols and c[1] >= 2*min(rows) and c[1] <= 2*max(rows)+2) or
                   (c[0] == 1 and c[4] in rows and c[1] >= 2*min(cols) and c[1] <= 2*max(cols)+2)]
    # (u, w, k, z) -> (u, w, k / 2, k % 2, z)

    subgraph = A.subgraph([dnx.zephyr_coordinates(gridsize).zephyr_to_linear(c)
                          for c in used_coords]).copy()

    return subgraph


def get_independent_embeddings(embs):
    """Finds a list of non-overlapping embeddings in `embs`.

    Args:
        embs (list[dict]): a list of embeddings (dict)

    Returns:
        List[dict]: a list of embeddings (dict)
    """
    start = time.process_time()

    Gemb = nx.Graph()
    Gemb.add_nodes_from(range(len(embs)))
    for i, emb1 in enumerate(embs):
        V1 = set(emb1.values())
        for j in range(i + 1, len(embs)):
            emb2 = embs[j]
            V2 = set(emb2.values())
            if not V1.isdisjoint(V2):
                Gemb.add_edge(i, j)
    print(f'Built graph.  Took {time.process_time()-start} seconds')
    start = time.process_time()

    Sbest = None
    max_size = 0
    for i in range(100000):
        if len(Gemb) > 0:
            S = nx.maximal_independent_set(Gemb)
        else:
            return []
        if len(S) > max_size:
            Sbest = S
            max_size = len(S)

    print(f'Built 100,000 greedy MIS.  Took {time.process_time()-start} seconds')
    print(f'Found {len(Sbest)} disjoint embeddings.')
    return [embs[x] for x in Sbest]


def search_for_subgraphs_in_subgrid(B, subgraph, timeout=20, max_number_of_embeddings=np.inf, verbose=True):
    """Find a list of subgraph (embeddings) in a subgrid.

    Args:
        B (nx.Graph): a subgrid
        subgraph (nx.Graph): subgraphs in B to search for
        timeout (int, optional): time limit for search. Defaults to 20.
        max_number_of_embeddings (int, optional): maximum number of embeddings to look for. Defaults to np.inf.
        verbose (bool, optional): Flag for verbosity. Defaults to True.

    Returns:
        List[dict]: a list of embeddings
    """
    embs = []
    while True and len(embs) < max_number_of_embeddings:
        temp = glasgow.find_subgraph(subgraph, B, timeout=timeout, triggered_restarts=True)
        if len(temp) == 0:
            break
        else:
            B.remove_nodes_from(temp.values())
            embs.append(temp)
            if verbose:
                print(f'{len(B)} vertices remain...')

    if verbose:
        print(f'Found {len(embs)} embeddings.')
    return embs


def raster_embedding_search(A_, subgraph, raster_breadth=5, delete_used=False,
                            verbose=True, topology='pegasus', gridsize=16, verify_embeddings=False, **kwargs):
    """Returns a matrix (n, L) of subgraph embeddings to _A.

    Args:
        _A (nx.Graph): target graph to embed to
        subgraph (nx.Graph): A smaller graph to embed into _A
        raster_breadth (int, optional): Breadth parameter of raster search. Defaults to 5.
        delete_used (bool, optional): Flag whether nodes in _A can appear in multiple embeddings.
                                      If set to true, nodes cannot be used in multiple embeddings. Defaults to False.
        verbose (bool, optional): Whether to print progress. Defaults to True.
        topology (str, optional): Name of topology. Defaults to 'pegasus'.
        gridsize (int, optional): Size of grid. Defaults to 16.
        verify_embeddings (bool, optional): Flag whether embeddings should be verified. Defaults to False.

    Raises:
        ValueError: Raise an error if given `topology` is unsupported
        Exception: Exception raised when embeddings are invalid and when `verify_embeddings` is True.

    Returns:
        numpy.ndarray: a matrix of embeddings
    """

    A = A_.copy()

    embs = []
    for row_offset in range(gridsize - raster_breadth + 1):

        for col_offset in range(gridsize - raster_breadth + 1):
            if topology == 'pegasus':
                B = get_pegasus_subgrid(
                    A, range(row_offset, row_offset + raster_breadth),
                    range(col_offset, col_offset + raster_breadth), gridsize
                )
            elif topology == 'zephyr':
                B = get_zephyr_subgrid(
                    A, range(row_offset, row_offset + raster_breadth),
                    range(col_offset, col_offset + raster_breadth), gridsize
                )
            else:
                raise ValueError("Supported topologies are currently pegasus and zephyr.")

            if verbose:
                print(f'row,col=({row_offset},{col_offset}) starting with {len(B)} vertices')

            sub_embs = search_for_subgraphs_in_subgrid(B, subgraph, verbose=verbose, **kwargs)
            if delete_used:
                for sub_emb in sub_embs:
                    A.remove_nodes_from(sub_emb.values())

            if verify_embeddings:
                for emb in sub_embs:
                    X = list(embedding.diagnose_embedding({p: [emb[p]] for p in emb}, subgraph, A_))
                    if len(X):
                        print(X[0])
                        raise Exception

            embs += sub_embs

    # Get independent set of embeddings
    independent_embs = get_independent_embeddings(embs)

    embmat = np.asarray([[ie[v] for ie in independent_embs] for v in subgraph.nodes]).T
    return embmat


def main():
    L = 64  # Length of chain to embed

    sampler = DWaveSampler()  # As configured
    bqm = dimod.BinaryQuadraticModel(vartype='SPIN')
    for spin in range(L):
        bqm.add_quadratic(spin, (spin + 1) % L, -1)
    G = dimod.to_networkx_graph(bqm)
    A = sampler.to_networkx_graph()

    embmat = raster_embedding_search(A, G, raster_breadth=3)


if __name__ == "__main__":
    main()
