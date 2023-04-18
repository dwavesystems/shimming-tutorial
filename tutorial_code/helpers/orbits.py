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
import numpy as np
import pynauty


def map_to_consecutive_integers(my_dict):
    """Utility function for mapping the values of a dict (these will be orbits) to consecutive integers 0,...,n-1

    Args:
        my_dict (dict): dictionary whose values are to be mapped to consecutive integers.

    Returns:
        dict: dictionary with keys in `my_dict`, and values in a range of consecutive integers
    """

    # Reduce coupler orbits to consecutive positive integers
    values = list(set(my_dict.values()))
    value_mapping = {values[i]: i for i in range(len(values))}
    ret = dict()
    for key in sorted(my_dict):
        ret[key] = value_mapping[my_dict[key]]
    return ret


def map_to_consecutive_integers_with_opposites(my_dict, opposites):
    """Utility function for mapping the values of a dict (these will be orbits) to consecutive integers 0,...,n-1
    while also handling opposite orbits.

    Args:
        my_dict (dict): dict of orbits
        opposites (dict): dict of opposite orbits

    Returns:
        Tuple[dict, dict]: a tuple of two dict of orbits with values mapped to consecutive integers
    """

    # Reduce coupler orbits to consecutive positive integers
    values = list(set(my_dict.values()))
    value_mapping = {values[i]: i for i in range(len(values))}
    ret = {}
    for key in sorted(my_dict):
        ret[key] = value_mapping[my_dict[key]]

    # Now the dict is correct, and we need to pad the value mapping with enough entries to cover the opposites.
    for i in list(opposites):
        if i not in value_mapping:
            value = max(list(value_mapping.values())) + 1
            value_mapping[i] = value

    retopp = opposites
    for i in list(set(ret.values())):
        retopp[i] = value_mapping[opposites[i]]

    return ret, retopp


def make_signed_bqm(bqm):
    """Takes a bqm and duplicates every spin s into two copies corresponding to s and -s.
       Each field h gets mapped to two opposing fields:
        h(s1) = -h(s2)
       each coupler gets mapped to four couplers:
        J(s1,s2) = J(-s1,-s2) = -J(s1,-s2) = -J(-s1,s2)

    Args:
        bqm (dimod.BQM): a binary quadratic model, i.e., Ising model under analysis

    Returns:
        dimod.BQM: a BQM representing a signed BQM
    """

    ret = dimod.BinaryQuadraticModel(vartype='SPIN')

    # Nodes and edges added in a seemingly ugly way in order to get the order right.
    for var in bqm.variables:
        ret.add_variable(f'p{var}', bqm.linear[var])
    for var in bqm.variables:
        ret.add_variable(f'm{var}', -bqm.linear[var])

    for (u, v) in bqm.quadratic:
        ret.add_quadratic(f'p{u}', f'p{v}', bqm.quadratic[(u, v)])
    for (u, v) in bqm.quadratic:
        ret.add_quadratic(f'm{u}', f'm{v}', bqm.quadratic[(u, v)])
    for (u, v) in bqm.quadratic:
        ret.add_quadratic(f'p{u}', f'm{v}', -bqm.quadratic[(u, v)])
    for (u, v) in bqm.quadratic:
        ret.add_quadratic(f'm{u}', f'p{v}', -bqm.quadratic[(u, v)])

    return ret


def get_bqm_orbits(bqm):
    """Takes a bqm, especially a "signed bqm" from `make_signed_bqm`, and converts it into a vertex-colored
    pynauty graph as needed, then gets orbits.

    Since pynauty only takes edge colorings, the couplings (J terms) need to be specified using
    auxiliary vertices.  Thus, for every edge (u,v) of the BQM graph, we add a new vertex w(u,v) and
    give it the color corresponding to J(u,v) in the BQM.

    To avoid ambiguity, we add a pendant (degree 1) vertex corresponding to each original vertex.

    Args:
        bqm (dimod.BQM): a binary quadratic model

    Returns:
        Tuple[dict, dict]: a tuple of qubit and coupler orbits
    """

    # The function first adds auxiliary elements to a BQM, then converts the BQM to a pynauty Graph()
    Gnx = nx.Graph()

    for v in bqm.variables:
        Gnx.add_node(f'hnode_{v}')
    for v in bqm.variables:
        Gnx.add_node(v)
        Gnx.add_edge(v, f'hnode_{v}')

    for (u, v) in bqm.quadratic:
        Gnx.add_edge(u, v)
        Gnx.add_node(f'Jnode_{u}_{v}')
        Gnx.add_edge(u, f'Jnode_{u}_{v}')
        Gnx.add_edge(v, f'Jnode_{u}_{v}')

    node_labels = list(Gnx.nodes)  # List of strings
    node_index_dict = {node_labels[i]: i for i in range(len(node_labels))}
    Gpn = pynauty.Graph(len(Gnx.nodes))

    for (u, v) in Gnx.edges:
        Gpn.connect_vertex(node_index_dict[u], node_index_dict[v])

    h_mapping = {h: [] for h in set(bqm.linear.values())}
    J_mapping = {J: [] for J in set(bqm.quadratic.values())}

    for p, q in bqm.linear.items():
        h_mapping[q].append(f'hnode_{p}')

    for p, q in bqm.quadratic.items():
        J_mapping[q].append(f'Jnode_{p[0]}_{p[1]}')

    # Make color classes
    coloring = []
    for V in h_mapping.values():
        coloring.append(
            set([node_index_dict[v] for v in V])
        )
    for E in J_mapping.values():
        coloring.append(
            set([node_index_dict[e] for e in E])
        )

    Gpn.set_vertex_coloring(coloring)

    # Orbits is just a vector of |V(Gpn)| numbers, each giving the canonical rep for its vertex orbit.
    ag = pynauty.autgrp(Gpn)[3]
    orbits = [int(x) for x in ag]

    qubit_orbits = {spin: orbits[node_index_dict[f'hnode_{spin}']] for spin in bqm.variables}
    coupler_orbits = {(u, v): orbits[node_index_dict[f'Jnode_{u}_{v}']] for u, v in bqm.quadratic}

    qubit_orbits = map_to_consecutive_integers(qubit_orbits)
    coupler_orbits = map_to_consecutive_integers(coupler_orbits)

    return qubit_orbits, coupler_orbits


def get_unsigned_bqm_orbits(signed_qubit_orbits, signed_coupler_orbits, bqm):
    """Assumes that orbits are given for a signed BQM, and turns them into signed orbits for an unsigned BQM.
    We also need to keep track of self-symmetric pairs of spins.

    Args:
        signed_qubit_orbits (dict): qubit orbits
        signed_coupler_orbits (dict): coupler orbits
        bqm (dimod.BQM): a binary quadratic model

    Returns:
        Tuple[dict, dict, dict, dict]: qubit_orbits, coupler_orbits, qubit_orbits_opposite, coupler_orbits_opposite
    """

    # Combine coupler orbits so that O(p1p2)=O(m1m2) and O(p1m2)=O(m1p2)
    for (u, v) in bqm.quadratic:
        signed_coupler_orbits[(f'p{u}', f'p{v}')] = min(
            signed_coupler_orbits[(f'p{u}', f'p{v}')],
            signed_coupler_orbits[(f'm{u}', f'm{v}')],
        )
        signed_coupler_orbits[(f'm{u}', f'm{v}')] = signed_coupler_orbits[(f'p{u}', f'p{v}')]

        signed_coupler_orbits[(f'm{v}', f'p{u}')] = min(
            signed_coupler_orbits[(f'm{v}', f'p{u}')],
            signed_coupler_orbits[(f'm{u}', f'p{v}')],
        )
        signed_coupler_orbits[(f'm{u}', f'p{v}')] = signed_coupler_orbits[(f'm{v}', f'p{u}')]

    signed_coupler_orbits = map_to_consecutive_integers(signed_coupler_orbits)

    qubit_orbits = {}
    coupler_orbits = {}

    # Get opposites
    qubit_orbits_opposite = [np.nan] * (1 + round(max(list(signed_qubit_orbits.values()))))
    coupler_orbits_opposite = [np.nan] * (1 + round(max(list(signed_coupler_orbits.values()))))

    for v in bqm.linear:
        qubit_orbits_opposite[signed_qubit_orbits[(f'p{v}')]] = signed_qubit_orbits[(f'm{v}')]
        qubit_orbits_opposite[signed_qubit_orbits[(f'm{v}')]] = signed_qubit_orbits[(f'p{v}')]
        qubit_orbits[v] = signed_qubit_orbits[(f'p{v}')]

    for (u, v) in bqm.quadratic:
        coupler_orbits_opposite[signed_coupler_orbits[(f'p{u}', f'p{v}')]] = (
            signed_coupler_orbits[(f'm{u}', f'p{v}')]
        )
        coupler_orbits_opposite[signed_coupler_orbits[(f'm{u}', f'p{v}')]] = (
            signed_coupler_orbits[(f'p{u}', f'p{v}')]
        )
        coupler_orbits[(u, v)] = signed_coupler_orbits[(f'p{u}', f'p{v}')]

    qubit_orbits, qubit_orbits_opposite = map_to_consecutive_integers_with_opposites(
        qubit_orbits, qubit_orbits_opposite)
    coupler_orbits, coupler_orbits_opposite = map_to_consecutive_integers_with_opposites(
        coupler_orbits, coupler_orbits_opposite)

    return qubit_orbits, coupler_orbits, qubit_orbits_opposite, coupler_orbits_opposite


def get_orbits(bqm):
    """Provide a binary quadratic model (BQM) and receive a set of usable orbits derived from the BQM.

    Args:
        bqm (dimod.BQM): a binary quadratic model

    Returns:
        Tuple[dict, dict, dict, dict]: (qubit_orbits, coupler_orbits, qubit_orbits_opposite, coupler_orbits_opposite)
    """

    bqm_ = dimod.BinaryQuadraticModel(vartype='SPIN')
    for var in bqm.variables:
        bqm_.add_variable(var, bqm.linear[var])
    for (u, v) in bqm.quadratic:
        bqm_.add_quadratic(u, v, bqm.quadratic[(u, v)])

    bqm = bqm_
    signed_bqm = make_signed_bqm(bqm)
    signed_qubit_orbits, signed_coupler_orbits = get_bqm_orbits(signed_bqm)
    return get_unsigned_bqm_orbits(signed_qubit_orbits, signed_coupler_orbits, bqm)


def to_networkx_graph(qubit_orbits, coupler_orbits):
    """Takes outputs of unsigned_bqm_orbits or get_bqm_orbits and makes a networkx graph whose linear and quadratic terms are
    qubit and coupler orbits.

    Args:
        qubit_orbits (dict): qubit orbits
        coupler_orbits (dict): coupler orbits

    Returns:
        nx.Graph: a networkx graph corresponding whose linear and quadratic terms are qubit and coupler orbits respectively.
    """

    Gnx = nx.Graph()
    Gnx.add_nodes_from(qubit_orbits)
    nx.set_node_attributes(Gnx, qubit_orbits, 'orbit')
    Gnx.add_edges_from(coupler_orbits)
    nx.set_edge_attributes(Gnx, coupler_orbits, 'orbit')

    return Gnx
