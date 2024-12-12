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
import networkx as nx

from matplotlib import pyplot as plt

from embed_square_lattice import make_square_bqm
from helpers import orbits
from helpers.helper_functions import get_qubit_colors, get_coupler_colors


def main(L=6, verbose=False):
    """Main function to run example.

    Visualizes an orbit calculation necessary for Figures 13-16 of DOI:10.3389/fcomp.2023.1238988
    a figure showing orbits of a LxL cylindrical square lattice is plotted.
    Note that vertex orbits are determined uniquely by distance from the cylinder
    top or bottom; reflecting the intuitive rotational and reflective
    symmetries. Edge orbits are similarly split by distance from the boundary
    and orientation relative to the boundary, but with additional splitting due
    to non-uniform coupling values. Edges and vertex in the same orbits have
    equivalent marginal statistics and can be shimmed to this effect.

    Args:
        L (int): Size of LxL square lattice, defaults to 6.
        verbose (bool): Print addition information on orbits to terminal. Defaults to False.
    """
    if L <= 2:
        raise ValueError("L>2 is required")
    bqm = make_square_bqm(L)
    # Get the orbits and opposite orbits for the BQM
    qubit_orbits, coupler_orbits, qubit_orbits_opposite, coupler_orbits_opposite = (
        orbits.get_orbits(bqm)
    )
    if verbose:
        # Print some information about the orbits.
        print("\nQubit orbits:")
        print(qubit_orbits)
        print("\nCoupler orbits:")
        print(coupler_orbits)
        print("")
        print("\nQubit orbits opposite:")
        for p, q in enumerate(qubit_orbits_opposite):
            print(f"QubitOrbit{p} = -QubitOrbit{q}")
        print("")
        print("\nCoupler orbits opposite:")
        for p, q in enumerate(coupler_orbits_opposite):
            print(f"CouplerOrbit{p} = -CouplerOrbit{q}")
        print("")

    Gnx = orbits.to_networkx_graph(qubit_orbits, coupler_orbits)
    plt.figure(figsize=(15, 8), dpi=80)
    options = {
        "node_size": 400,
        "width": 4,
    }
    plt.gcf().canvas.manager.set_window_title(
        "Orbit Visualization related to Figure 13 - 16"
    )
    pos = nx.spring_layout(Gnx, iterations=500, dim=2)  # 2D spring layout
    nx.draw(
        Gnx,
        pos=pos,
        node_color=get_qubit_colors(Gnx, bqm),
        edge_color=get_coupler_colors(Gnx, bqm),
        **options,
    )
    node_labels = {
        key: f"{val}" for key, val in nx.get_node_attributes(Gnx, "orbit").items()
    }
    nx.draw_networkx_labels(Gnx, pos=pos, labels=node_labels, font_size=14)
    edge_labels = {
        key: f"{val}" for key, val in nx.get_edge_attributes(Gnx, "orbit").items()
    }
    nx.draw_networkx_edge_labels(Gnx, pos=pos, edge_labels=edge_labels, font_size=14)

    # A printed comment is appropriate since there is no figure to refer to
    print(
        "Visualizes an orbit calculation necessary for Figures 13-16 of DOI:10.3389/fcomp.2023.1238988"
        "\na figure showing orbits of a LxL cylindrical square lattice is plotted."
        "\nNote that vertex orbits are determined uniquely by distance from the cylinder"
        "\ntop or bottom; reflecting the intuitive rotational and reflective"
        "\nsymmetries. Edge orbits are similarly split by distance from the boundary"
        "\nand orientation relative to the boundary, but with additional splitting due"
        "\nto non-uniform coupling values. Edges and vertex in the same orbits have"
        "\nequivalent marginal statistics and can be shimmed to this effect."
    )

    plt.show()


if __name__ == "__main__":
    main()
