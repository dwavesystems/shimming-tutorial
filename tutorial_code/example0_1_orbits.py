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
import networkx as nx

from matplotlib import pyplot as plt

from helpers.helper_functions import get_coupler_colors, get_qubit_colors
from helpers import orbits


def make_bqm():
    """Makes a simple four-spin BQM

    Returns:
        dimod.BQM: a binary quadratic model with four spins
    """
    bqm = dimod.BinaryQuadraticModel(vartype='SPIN')

    for x in range(4):
        bqm.add_variable(x)
    bqm.set_linear(0, 1)
    bqm.set_quadratic(0, 1, 1.)
    bqm.set_quadratic(1, 2, -1.)
    bqm.set_quadratic(2, 3, -1.)
    bqm.set_quadratic(3, 0, -1.)

    return bqm


def main():
    """Main function to run example
    """
    # Make the four-qubit BQM and compute its orbits.
    bqm = make_bqm()
    signed_bqm = orbits.make_signed_bqm(bqm)
    signed_qubit_orbits, signed_coupler_orbits = orbits.get_bqm_orbits(signed_bqm)
    qubit_orbits, coupler_orbits, qubit_orbits_opposite, coupler_orbits_opposite = \
        orbits.get_orbits(bqm)

    # Print some information about the signed orbits (orbits of the signed BQM).
    print('Signed qubit orbits:')
    print(signed_qubit_orbits)

    print('\nSigned coupler orbits:')
    print(signed_coupler_orbits)

    # Print some information about the orbits.
    print('\nQubit orbits:')
    print(qubit_orbits)
    print('\nCoupler orbits:')
    print(coupler_orbits)
    print('')
    print('\nQubit orbits opposite:')
    for p, q in enumerate(qubit_orbits_opposite):
        print(f'QubitOrbit{p} = -QubitOrbit{q}')
    print('')
    print('\nCoupler orbits opposite:')
    for p, q in enumerate(coupler_orbits_opposite):
        print(f'CouplerOrbit{p} = -CouplerOrbit{q}')
    print('')

    qubit_orbit_sizes = [len([x for x in qubit_orbits.values() if int(x) == i]) for i in
                         range(len(qubit_orbits_opposite))]
    coupler_orbit_sizes = [len([x for x in coupler_orbits.values() if int(x) == i]) for i in
                           range(len(coupler_orbits_opposite))]

    for i in range(len(qubit_orbit_sizes)):
        if qubit_orbit_sizes[i] > 0 and qubit_orbit_sizes[qubit_orbits_opposite[i]] > 0:
            print(f'Qubit orbit {i} has opposite {qubit_orbits_opposite[i]}.')

    for i in range(len(coupler_orbit_sizes)):
        if coupler_orbit_sizes[i] > 0 and coupler_orbit_sizes[coupler_orbits_opposite[i]] > 0:
            print(f'Coupler orbit {i} has opposite {coupler_orbits_opposite[i]}.')

    # Now we plot the BQM (h,J), the signed BWM S(h,J), and their qubit and coupler orbits.
    plt.rc('font', size=12)
    plt.figure(figsize=(11, 5), dpi=80)
    options = {'node_size': 600,
               'width': 4,
               }

    Gnx = orbits.to_networkx_graph(qubit_orbits, coupler_orbits)
    pos = {0: [-1 - 4, 1], 1: [1 - 4, 1], 2: [1 - 4, -1], 3: [-1 - 4, -1]}
    nx.draw(Gnx, pos=pos,
            edge_color=get_coupler_colors(Gnx, bqm),
            node_color=get_qubit_colors(Gnx, bqm),
            **options)
    node_labels = {key: f'{val}' for key, val in nx.get_node_attributes(Gnx, "orbit").items()}
    nx.draw_networkx_labels(Gnx, pos=pos, labels=node_labels, font_size=14)
    edge_labels = {key: f'{val}' for key, val in nx.get_edge_attributes(Gnx, "orbit").items()}
    nx.draw_networkx_edge_labels(Gnx, pos=pos, edge_labels=edge_labels, font_size=14)

    Gnx = orbits.to_networkx_graph(signed_qubit_orbits, signed_coupler_orbits)
    pos = {
        'p0': [-1, 1], 'p1': [1, 1], 'p2': [1, -1], 'p3': [-1, -1],
        'm0': [-1.5, 1.5], 'm1': [1.5, 1.5], 'm2': [1.5, -1.5], 'm3': [-1.5, -1.5],
    }
    nx.draw(Gnx, pos=pos,
            edge_color=get_coupler_colors(Gnx, signed_bqm),
            node_color=get_qubit_colors(Gnx, signed_bqm),
            **options)
    node_labels = {key: f'{val}' for key, val in nx.get_node_attributes(Gnx, "orbit").items()}
    nx.draw_networkx_labels(Gnx, pos=pos, labels=node_labels, font_size=14)
    edge_labels = {key: f'{val}' for key, val in nx.get_edge_attributes(Gnx, "orbit").items()}
    nx.draw_networkx_edge_labels(Gnx, pos=pos, edge_labels=edge_labels, font_size=14)
    plt.show()


if __name__ == "__main__":
    main()
