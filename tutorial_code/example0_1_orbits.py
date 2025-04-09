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

import networkx as nx
from matplotlib import pyplot as plt

import dimod

from helpers.helper_functions import get_coupler_colors, get_qubit_colors
from helpers import orbits


def plot_orbits(Gnx: nx.Graph, pos: dict, bqm: dimod.BinaryQuadraticModel):
    """Plot orbits associated to graphs."""
    options = {"node_size": 600, "width": 4}
    nx.draw(
        Gnx,
        pos=pos,
        edge_color=get_coupler_colors(Gnx, bqm=bqm),
        node_color=get_qubit_colors(Gnx, bqm=bqm),
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


def make_bqm():
    """Creates a simple model of four connected spins (a Binary Quadratic Model).

    The BQM model represents four spins that interact with their neighbors.
    We define how these spins are influenced (called 'fields') and how they interact (called 'couplings').

    Returns:
        dimod.BQM: A binary quadratic model with four spins and specific coupling between them
    """
    bqm = dimod.BinaryQuadraticModel(vartype="SPIN")

    for x in range(4):
        bqm.add_variable(x)

    bqm.set_linear(0, 1)
    bqm.set_quadratic(0, 1, 1.0)
    bqm.set_quadratic(1, 2, -1.0)
    bqm.set_quadratic(2, 3, -1.0)
    bqm.set_quadratic(3, 0, -1.0)

    return bqm


def main(verbose: bool = False) -> None:
    """Reproduces Figure 4 of the shimming tutorial DOI:10.3389/fcomp.2023.1238988

    Example of creating a 4-variable Binary Quadratic Model (BQM)
    and computing its orbits (symmetries of spins and couplers).

    Args:
        verbose (bool): Print addition information on orbits to terminal. Defaults to False.
    """

    bqm = make_bqm()
    signed_bqm = orbits.make_signed_bqm(bqm)
    signed_qubit_orbits, signed_coupler_orbits = orbits.get_bqm_orbits(signed_bqm)
    qubit_orbits, coupler_orbits, qubit_orbits_opposite, coupler_orbits_opposite = (
        orbits.get_orbits(bqm)
    )
    if verbose:
        print(f"Signed qubit orbits: {signed_qubit_orbits}")
        print(f"\nSigned coupler orbits: {signed_coupler_orbits}")
        print(f"\nQubit orbits: {qubit_orbits}")
        print(f"\nCoupler orbits: {coupler_orbits}")

        print("\nQubit orbits opposite:")
        for p, q in enumerate(qubit_orbits_opposite):
            print(f"QubitOrbit{p} = -QubitOrbit{q}")

        print("\nCoupler orbits opposite:")
        for p, q in enumerate(coupler_orbits_opposite):
            print(f"CouplerOrbit{p} = -CouplerOrbit{q}")

    qubit_orbit_sizes = [
        len([x for x in qubit_orbits.values() if int(x) == i])
        for i in range(len(qubit_orbits_opposite))
    ]
    coupler_orbit_sizes = [
        len([x for x in coupler_orbits.values() if int(x) == i])
        for i in range(len(coupler_orbits_opposite))
    ]
    if verbose:
        print(f"Qubit orbit sizes: {qubit_orbit_sizes}")
        print(f"Coupler orbit sizes: {coupler_orbit_sizes}")

        for i in range(len(qubit_orbit_sizes)):
            if (
                qubit_orbit_sizes[i] > 0
                and qubit_orbit_sizes[qubit_orbits_opposite[i]] > 0
            ):
                print(
                    f"Qubit orbit {i} has behavior opposite to qubit orbit {qubit_orbits_opposite[i]}."
                )

        for i in range(len(coupler_orbit_sizes)):
            if (
                coupler_orbit_sizes[i] > 0
                and coupler_orbit_sizes[coupler_orbits_opposite[i]] > 0
            ):
                print(
                    f"Coupler orbit {i} behavior opposite to coupler orbit {coupler_orbits_opposite[i]}."
                )

    plt.rc("font", size=12)
    fig = plt.figure(figsize=(11, 5), dpi=80)
    fig.canvas.manager.set_window_title(
        "Figure 4: Orbits of signed and original Ising model"
    )

    Gnx = orbits.to_networkx_graph(qubit_orbits, coupler_orbits)
    pos = {0: [-1 - 4, 1], 1: [1 - 4, 1], 2: [1 - 4, -1], 3: [-1 - 4, -1]}

    plot_orbits(Gnx=Gnx, pos=pos, bqm=bqm)

    Gnx = orbits.to_networkx_graph(signed_qubit_orbits, signed_coupler_orbits)
    pos = {
        "p0": [-1, 1],
        "p1": [1, 1],
        "p2": [1, -1],
        "p3": [-1, -1],
        "m0": [-1.5, 1.5],
        "m1": [1.5, 1.5],
        "m2": [1.5, -1.5],
        "m3": [-1.5, -1.5],
    }

    plot_orbits(Gnx=Gnx, pos=pos, bqm=signed_bqm)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="example0_1_orbits")
    parser.add_argument(
        "--verbose", action="store_true", help="Print additional verbose information"
    )
    args = parser.parse_args()
    main(verbose=args.verbose)
