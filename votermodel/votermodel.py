#-*- coding: utf-8 -*-
from __future__ import (print_function, division, 
                        absolute_import, unicode_literals)

import networkx as nx
import numpy as np
import random

class VoterModel(object):
    """Voter model.

    Wikipedia:
    https://en.wikipedia.org/wiki/Voter_model

    """
    def __init__(self, network):
        """Initialization method.

        Parameters
        ----------
            network : networkx Graph instance

        """
        self.network = nx.convert_node_labels_to_integers(network)
        self.nnodes = network.number_of_nodes()
        # Array to store the node state
        self.states = np.random.randint(2, size=self.nnodes)

    def meanstate(self):
        return np.average(self.states)

    def interface_dens(self):
        """Calculate the interface density of the system.

        """
        n_interfaces = 0

        # Calculate the number of active interfaces
        for edge in self.network.edges():
            if self.states[edge[0]] != self.states[edge[1]]:
                n_interfaces += 1

        return float(n_interfaces)/self.network.number_of_edges()

    
    def update(self, nsteps):
        """Update the system.

        Parameters
        ----------
            nsteps : int
                Number of Monte Carlo (MC) steps. One MC step is made
                of nnodes changes.

        """
        for n in range(nsteps*self.nnodes):
            # Choose node randomly
            node = np.random.randint(self.nnodes)
            neighbors = self.network.neighbors(node)
            # TODO: DELETE
            # Copy random neighbour state
            self.states[node] = self.states[random.choice(list(neighbors))]
            
