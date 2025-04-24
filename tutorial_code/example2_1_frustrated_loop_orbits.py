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
import argparse

import matplotlib
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import dimod

from helpers import paper_plotting_functions, orbits


def get_vertex_coordinates(Gnx, L, num_loops):
    """Get vertex coordinates for plotting in matplotlib.

    Args:
        Gnx (nx.Graph): the graph to plot
        L (int): length of frustrated loops
        num_loops (int): number of frustrated loops

    Returns:
        dict: a dictionary of positions keyed by node
    """
    pos = nx.kamada_kawai_layout(Gnx)
    # Adjust node positions
    spacing = 1.2 * (
        max([pos[v][0] for v in range(L)]) - min([pos[v][0] for v in range(L)])
    )
    for i in range(L):
        for replica in range(1, num_loops):
            pos[i + replica * L] = pos[i] + np.array([spacing, 0]) * replica

    return pos


def get_edge_colors(Gnx, bqm):
    """Generate a list of edge colours given a graph and BQM

    Args:
        Gnx (nx.Graph): the graph to plot
        bqm (dimod.BinaryQuadraticModel): the BQM to visualize

    Returns:
        List[tuple[float, float, float, float]]: a list of tuple of RGBA values
    """
    cm = matplotlib.colormaps["RdBu_r"]
    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
    return [cm(norm(bqm.get_quadratic(u, v))) for (u, v) in Gnx.edges()]


def main(L=6, num_loops=3, verbose=False):
    """Main function to run example.

    Completes an experiment matched to Figure 9 of DOI:10.3389/fcomp.2023.1238988,
    plotting a corresponding figure.

    Args:
        L (int): Length of the frustrated loop.
        num_loops (int): Number of independent rings.
        verbose (bool): Print addition information on orbits to terminal. Defaults to False.
    """
    bqm = dimod.BinaryQuadraticModel(
        vartype="SPIN",
    )
    for iL in range(num_loops):
        for spin in range(L - 1):
            bqm.add_quadratic(spin + L * iL, ((spin + 1) % L) + L * iL, -1.0)
        bqm.add_quadratic(L - 1 + L * iL, 0 + L * iL, 1.0)

    # Get the orbits and opposite orbits for the BQM
    qubit_orbits, coupler_orbits, qubit_orbits_opposite, coupler_orbits_opposite = (
        orbits.get_orbits(bqm)
    )
    if verbose:
        # Print some information about the orbits.
        print(f"\nQubit orbits: {qubit_orbits}")
        print(f"\nCoupler orbits: {coupler_orbits}")
        print("")
        print("\nQubit orbits opposite:")
        for p, q in enumerate(qubit_orbits_opposite):
            print(f"QubitOrbit{p} = -QubitOrbit{q}")
        print("")
        print("\nCoupler orbits opposite:")
        for p, q in enumerate(coupler_orbits_opposite):
            print(f"CouplerOrbit{p} = -CouplerOrbit{q}")
        print("")

    # Make a networkx whose coupler orbits are indicated on the edges, for drawing.
    Gnx = orbits.to_networkx_graph(qubit_orbits, coupler_orbits)

    # Draw the graph
    fig = plt.figure()
    fig.canvas.manager.set_window_title("Figure 9: Coupler orbits of frustrated loops")
    options = {
        "node_color": np.atleast_2d([0.8, 0.8, 0.8]),
        "node_size": 400,
        "width": 4,
    }
    pos = get_vertex_coordinates(Gnx, L, num_loops)
    nx.draw(Gnx, pos=pos, edge_color=get_edge_colors(Gnx, bqm), **options)

    # Draw the node labels
    node_labels = {
        key: f"{val}" for key, val in nx.get_node_attributes(Gnx, "orbit").items()
    }
    nx.draw_networkx_labels(Gnx, pos=pos, labels=node_labels)

    # Draw the edge labels, indicating the orbit of each edge (coupler)
    nx.draw_networkx_edge_labels(
        Gnx,
        pos=pos,
        edge_labels={
            key: f"{int(val)}"
            for key, val in nx.get_edge_attributes(Gnx, "orbit").items()
        },
    )

    paper_plotting_functions.paper_plots_example2_1()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="example2_1_frustrated_loop_orbits")
    parser.add_argument(
        "--verbose", action="store_true", help="Print additional verbose information"
    )
    parser.add_argument(
        "--L", default=6, type=int, help="Length of the loop"
    )
    parser.add_argument(
        "--num_loops", default=3, type=int, help="Number of independent rings"
    )
    args = parser.parse_args()
    main(L=args.L, num_loops=args.num_loops, verbose=args.verbose)
