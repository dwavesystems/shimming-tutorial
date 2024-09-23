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
    """
    
    Creates a simple model of four connected spins (a Binary Quadratic Model)

    The BQM model represents four "spns" that interact with four spins that interact with their neighbors.
    We define how these spins are influenced (called 'fields') and how they interact (called 'couplings').

    Returns:
        dimod.BQM: a binary quadratic model with four spins and specific coupling between them
    """
    bqm = dimod.BinaryQuadraticModel(vartype='SPIN')

    # Add 4 spins to the model 
    for x in range(4):
        bqm.add_variable(x)
    
    # Setting the fields and couplings (how spins interact)
    bqm.set_linear(0, 1) # Spin 0 is influenced with a field of +1
    bqm.set_quadratic(0, 1, 1.) # Spin 0 and Spin 1 interact positively
    bqm.set_quadratic(1, 2, -1.) # Spin 1 and Spin 2 interact negaively 
    bqm.set_quadratic(2, 3, -1.) # Spin 2 and Spin 3 interact negatively
    bqm.set_quadratic(3, 0, -1.) # Spin 3 and Spin 0 interact negatively 

    return bqm

# What is a Binary Quadratic Model (BQM)?
# In simple terms, a BQM is a mathematical model used to represent problems where we have variables that
# can be +1 or -1 (like tiny magnets called "spins") and these spins interact with each other.
# The goal is to find the lowest energy state, where the interactions between spins are satisfied as much as possible.

# What are Qubit Orbits and Coupler Orbits?
# Qubit orbits group spins that behave similarly based on system symmetry. This allows us to treat certain
# spins as identical, making the system easier to analyze. Coupler orbits group the interactions (connections)
# between spins that behave similarly.
def main():
    """
    Main function to run the example of creating a Binary Quadratic Model (BQM) 
    and computing its orbits (symmetries of spins and couplers).

    """
    # Make the four-qubit BQM
    try:
        bqm = make_bqm()
    except Exception as e: 
        print("An error occured while creating the model. Please double check the input data.")

    # Create a signed version of the BQM, duplicating the spins and couplings with negations
    signed_bqm = orbits.make_signed_bqm(bqm)

    # Compute the orbits (symmetries) of the signed BQM
    signed_qubit_orbits, signed_coupler_orbits = orbits.get_bqm_orbits(signed_bqm)

    # Compute the orbits for the original BQM
    qubit_orbits, coupler_orbits, qubit_orbits_opposite, coupler_orbits_opposite = \
        orbits.get_orbits(bqm)

    # Print information about the signed qubit orbits (how similar spins behave)
    print('Signed qubit orbits:')
    print(signed_qubit_orbits)

    # Print information about the signed coupler orbits (how similar couplings behave)
    print('\nSigned coupler orbits:')
    print(signed_coupler_orbits)

    # Print information about the qubit orbits for the original BQM
    print('\nQubit orbits:')
    print(qubit_orbits)
    
    # Print information about the coupler orbits for the original BQM
    print('\nCoupler orbits:')
    print(coupler_orbits)

    # Show the opposite relationships between qubit orbits
    print('\nQubit orbits opposite:')
    for p, q in enumerate(qubit_orbits_opposite):
        print(f'QubitOrbit{p} = -QubitOrbit{q}')

    # Show the opposite relationships between coupler orbits
    print('\nCoupler orbits opposite:')
    for p, q in enumerate(coupler_orbits_opposite):
        print(f'CouplerOrbit{p} = -CouplerOrbit{q}')

    # Calculate the size of each qubit orbit and its opposite
    qubit_orbit_sizes = [len([x for x in qubit_orbits.values() if int(x) == i]) for i in
                         range(len(qubit_orbits_opposite))]
    coupler_orbit_sizes = [len([x for x in coupler_orbits.values() if int(x) == i]) for i in
                           range(len(coupler_orbits_opposite))]
    
    print(f'Qubit orbit sizes: {qubit_orbit_sizes}')
    print(f'Coupler orbit sizes: {coupler_orbit_sizes}')


    print('')
     # Print information about qubit orbits and their opposite relationships based on sizes
    for i in range(len(qubit_orbit_sizes)):
        if qubit_orbit_sizes[i] > 0 and qubit_orbit_sizes[qubit_orbits_opposite[i]] > 0:
            print(f'Qubit orbit {i} has behavior opposite to qubit orbit {qubit_orbits_opposite[i]}.')

    print('')
    # Print information about coupler orbits and their opposite relationships based on sizes
    for i in range(len(coupler_orbit_sizes)):
        if coupler_orbit_sizes[i] > 0 and coupler_orbit_sizes[coupler_orbits_opposite[i]] > 0:
            print(f'Coupler orbit {i} behavior opposite to coupler orbit {coupler_orbits_opposite[i]}.')

    # Now, we plot the Binary Quadratic Model (BQM) and its signed version
    plt.rc('font', size=12)
    plt.figure(figsize=(11, 5), dpi=80)  # Create a figure for the plots
    options = {'node_size': 600, 'width': 4,}  # Set options for plot visuals

    # Plot the original BQM (showing qubits and couplers)
    Gnx = orbits.to_networkx_graph(qubit_orbits, coupler_orbits)
    pos = {0: [-1 - 4, 1], 1: [1 - 4, 1], 2: [1 - 4, -1], 3: [-1 - 4, -1]}  # Position the qubits
    nx.draw(Gnx, pos=pos,
            edge_color=get_coupler_colors(Gnx, bqm),
            node_color=get_qubit_colors(Gnx, bqm),
            **options)
    
    # Add labels for the qubits (spins) and couplers
    node_labels = {key: f'{val}' for key, val in nx.get_node_attributes(Gnx, "orbit").items()}
    nx.draw_networkx_labels(Gnx, pos=pos, labels=node_labels, font_size=14)
    edge_labels = {key: f'{val}' for key, val in nx.get_edge_attributes(Gnx, "orbit").items()}
    nx.draw_networkx_edge_labels(Gnx, pos=pos, edge_labels=edge_labels, font_size=14)

    # Plot the signed version of the BQM (showing duplicated qubits and couplers)
    Gnx = orbits.to_networkx_graph(signed_qubit_orbits, signed_coupler_orbits)
    pos = {
        'p0': [-1, 1], 'p1': [1, 1], 'p2': [1, -1], 'p3': [-1, -1],
        'm0': [-1.5, 1.5], 'm1': [1.5, 1.5], 'm2': [1.5, -1.5], 'm3': [-1.5, -1.5],
    } # Position for signed BQM
    nx.draw(Gnx, pos=pos,
            edge_color=get_coupler_colors(Gnx, signed_bqm),
            node_color=get_qubit_colors(Gnx, signed_bqm),
            **options)
    
    # Add labels for signed qubits and couplers
    node_labels = {key: f'{val}' for key, val in nx.get_node_attributes(Gnx, "orbit").items()}
    nx.draw_networkx_labels(Gnx, pos=pos, labels=node_labels, font_size=14)
    edge_labels = {key: f'{val}' for key, val in nx.get_edge_attributes(Gnx, "orbit").items()}
    nx.draw_networkx_edge_labels(Gnx, pos=pos, edge_labels=edge_labels, font_size=14)

    # Show the final plot
    plt.show()

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
