# Measure Barabasi-Albert network

import os
import sys
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import votermodel as voter

# Parameters
# ====================
nnodes = 200
edges_per_node_list = np.array([2, 10, 50])
sims_per_prob = 100
nsteps = 600

# Measure
meanstates = np.zeros((len(edges_per_node_list), nsteps), dtype=float)
interfacedens = np.zeros((len(edges_per_node_list), nsteps), dtype=float)

for j_edges, edges_per_node in enumerate(edges_per_node_list):
    meanstates_sum = np.zeros(nsteps, dtype=float)
    interfacedens_sum = np.zeros(nsteps, dtype=float)
    for j_sim in range(sims_per_prob): 
        print("N edges: {0} ({1}/{2}), sim {3}/{4}".format(
                edges_per_node, j_edges + 1, len(edges_per_node_list), j_sim + 1,
                sims_per_prob))
        # Initialize instance
        net = nx.barabasi_albert_graph(nnodes, edges_per_node)
        VM = voter.VoterModel(net)

        for j_step in range(nsteps):
            VM.update(1)
            meanstates_sum[j_step] += np.abs(VM.meanstate() - 0.5)/0.5
            interfacedens_sum[j_step] += VM.interface_dens()

    # Calculate and store mean
    meanstates[j_edges] = meanstates_sum/sims_per_prob
    interfacedens[j_edges] = interfacedens_sum/sims_per_prob


# Save data
data_meanstate = np.hstack((edges_per_node_list[np.newaxis].T, meanstates))
np.savetxt("BA_meanstates.dat", data_meanstate)

data_interfacedens = np.hstack((edges_per_node_list[np.newaxis].T, interfacedens))
np.savetxt("BA_interfacedens.dat", data_interfacedens)


# Plot results
size = 3
fig, ax = plt.subplots(figsize=(1.62*size, size))
#ax.set_ylabel(r"$(\langle s \rangle - 0.5)/0.5$")
#
#for j_edges, edges_per_node in enumerate(edges_per_node_list):
#    ax.plot(meanstates[j_edges], label=r"$m={0}$".format(edges_per_node))

ax.set_xlabel(r"Time")
ax.set_ylabel(r"Interface density")

for j_edges, edges_per_node in enumerate(edges_per_node_list):
    ax.plot(interfacedens[j_edges], label=r"$m={0}$".format(edges_per_node))

ax.legend()
fig.tight_layout()

fig.savefig("BA.png", dpi=200)
#plt.show()
