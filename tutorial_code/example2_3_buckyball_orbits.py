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
from os.path import exists

import dimod
import networkx as nx
import numpy as np

from dwave.system import DWaveSampler
from matplotlib import pyplot as plt

from helpers import orbits
from helpers.embedding_helpers import raster_embedding_search
from helpers.orbits import get_orbits


def main(visualize=True):
    """Main function to run example

    Args:
        visualize (bool, optional): flag for visualization. Defaults to True.
    """
    # Parse the BQM
    path_to_csv = "data/bucky_ball.csv"
    if not exists(path_to_csv):
        path_to_csv = f"tutorial_code/{path_to_csv}"
    J = {(int(e[0]), int(e[1])): w
         for *e, w in np.loadtxt(path_to_csv, delimiter=",")}
    bqm = dimod.BQM.from_ising(h={}, J=J)

    # Compute the BQM's orbits
    (qubit_orbits, coupler_orbits,
        qubit_orbits_opposite, coupler_orbits_opposite) = get_orbits(bqm)

    # Embed the BQM onto the QPU
    qpu = DWaveSampler(solver="Advantage_system4.1")
    graph_qpu = qpu.to_networkx_graph()
    graph_bqm = dimod.to_networkx_graph(bqm)

    embeddings = raster_embedding_search(graph_qpu, graph_bqm, raster_breadth=2)
    print(embeddings)

    if visualize:
        # Plotting configurations
        cm = plt.cm.get_cmap(name='coolwarm')
        norm = plt.Normalize(vmin=-2, vmax=2)
        plt.rc('font', size=12)

        fig = plt.figure(figsize=(8, 8), dpi=80)

        orbits_graph = orbits.to_networkx_graph(qubit_orbits, coupler_orbits)
        edge_color = [cm(norm(1)) if orbits_graph[u][v]['orbit'] else cm(norm(-1))
                      for u, v in orbits_graph.edges]
        node_color = [cm(norm(qubit_orbits[u])) for u in orbits_graph.nodes]
        pos = nx.layout.spring_layout(orbits_graph, iterations=10000, seed=5)
        nx.draw_networkx(orbits_graph, pos=pos, with_labels=False, node_size=200, width=4,
                         edge_color=edge_color, node_color=node_color,)
        plt.axis("off")
        fig.savefig("buckyball_orbits.pdf", format="pdf")


if __name__ == "__main__":
    main()
