# Measure Watts-Strogatz network

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
edges_per_node_BA = 2
edges_per_node_WS = 4
rewire_probs = np.array([0, 0.001, 0.01, 0.1, 1])
sims_per_prob =  10
nsteps = 1000


# Measure
meanstates = np.zeros((len(rewire_probs), nsteps), dtype=float)
interfacedens = np.zeros((len(rewire_probs), nsteps), dtype=float)

for j_prob, rewire_prob in enumerate(rewire_probs):
    meanstates_sum = np.zeros(nsteps, dtype=float)
    interfacedens_sum = np.zeros(nsteps, dtype=float)
    for j_sim in range(sims_per_prob): 
        print("Prob: {0} ({1}/{2}), Sim: {3}/{4}".format(
                rewire_prob, j_prob + 1, len(rewire_probs), j_sim + 1,
                sims_per_prob))
        # Initialize instance
        net_WS = nx.watts_strogatz_graph(nnodes, edges_per_node_WS, rewire_prob)
        VM_WS = voter.VoterModel(net_WS)

        for j_step in range(nsteps):
            VM_WS.update(1)
            meanstates_sum[j_step] += np.abs(VM_WS.meanstate() - 0.5)/0.5
            interfacedens_sum[j_step] += VM_WS.interface_dens()

    # Calculate and store mean
    meanstates[j_prob] = meanstates_sum/sims_per_prob
    interfacedens[j_prob] = interfacedens_sum/sims_per_prob


# Save data
data_meanstate = np.hstack((rewire_probs[np.newaxis].T, meanstates))
np.savetxt("WS_meanstates.dat", data_meanstate)

data_interfacedens = np.hstack((rewire_probs[np.newaxis].T, interfacedens))
np.savetxt("WS_interfacedens.dat", data_interfacedens)

# Plot results
size = 3
fig, ax = plt.subplots(figsize=(1.62*size, size))
#ax.set_ylabel(r"$(\langle s \rangle - 0.5)/0.5$")
#
#for j_prob, rewire_prob in enumerate(rewire_probs):
#    ax.plot(meanstates_WS[j_prob], label=r"$p={0}$".format(rewire_prob))

ax.set_xlabel(r"Time")
ax.set_ylabel(r"Interface density")

for j_edges, rewire_prob in enumerate(rewire_probs):
    ax.plot(interfacedens[j_edges], label=r"$m={0}$".format(rewire_prob))

ax.legend()
fig.tight_layout()

fig.savefig("WS_p.png", dpi=200)
#plt.show()
