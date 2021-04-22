""" Sequential uncertainty reduction on excursion sets. Class defining the
weighted IVR acquisition function.

"""
from volcapy.strategy.strategyABC import StrategyABC
import numpy as np


class MyopicWIVRStrategy(StrategyABC):
    def get_next_ind(self):
        # Evaluate criterion on neighbors.
        neighbors_inds = self.get_neighbors(self.current_ind)
        neighbors_ivrs = []

        # Compute criterion for each candidate.
        for ind in neighbors_inds:
            # Observation operator for candidate location.
            candidate_G = self.G[ind,:].reshape(1, -1)
            ivr = self.gp.IVR(candidate_G, data_std, 
                                weights=self.current_coverage)
            neighbors_ivrs.append(ivr)

        # Find best candidate.
        tmp_ind = np.argmax(neighbors_ivrs)
        best_ind = neighbors_inds[tmp_ind]
        return best_ind
