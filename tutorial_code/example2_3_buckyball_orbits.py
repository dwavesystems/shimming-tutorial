import dimod
import numpy as np

from helpers.orbits import get_orbits

from dwave.system import DWaveSampler

from minorminer import find_embedding


if __name__ == "__main__":
    def main(visualize=True):
        # Parse the BQM
        J = {(int(e[0]), int(e[1])): w
             for *e, w in np.loadtxt("tutorial_code/data/bucky_ball.csv", delimiter=",")}
        bqm = dimod.BQM.from_ising(h={}, J=J)

        # Compute the BQM's orbits
        (qubit_orbits, coupler_orbits,
         qubit_orbits_opposite, coupler_orbits_opposite) = get_orbits(bqm)

        # Embed the BQM onto the QPU
        qpu = DWaveSampler(solver="Advantage_system4.1")
        graph_qpu = qpu.to_networkx_graph()
        graph_bqm = dimod.to_networkx_graph(bqm)
        # Here we use an off-the-shelf heuristic method to find embeddings of the Ising model to the QPU
        embedding = find_embedding(graph_bqm, graph_qpu)

        if visualize:
            import networkx as nx
            import matplotlib.pyplot as plt
            from helpers import orbits

            # Plotting configurations
            cm = plt.cm.get_cmap(name='coolwarm')
            norm = plt.Normalize(vmin=-2, vmax=2)
            plt.rc('font', size=12)

            fig = plt.figure(figsize=(8, 8), dpi=80)

            orbits_graph = orbits.to_networkx_graph(qubit_orbits, coupler_orbits)
            edge_color = [cm(norm(1)) if orbits_graph[u][v]['orbit'] else cm(norm(-1)) for u, v in orbits_graph.edges]
            node_color = [cm(norm(qubit_orbits[u])) for u in orbits_graph.nodes]
            pos = nx.layout.spring_layout(orbits_graph, iterations=10000, seed=5)
            nx.draw_networkx(orbits_graph, pos=pos, with_labels=False, node_size=200, width=4, edge_color=edge_color, node_color=node_color,)
            plt.axis("off")
            fig.savefig("buckyball_orbits.pdf", format="pdf")
        return
    main()

print("Exited")
