""" Sequential uncertainty reduction on excursion sets. Class defining the
rollout version of the weighted IVR acquisition function.

"""
from volcapy.strategy.strategyABC import StrategyABC
import numpy as np


class RolloutWIVRStrategy(StrategyABC):
    def get_next_ind(self):
        # Evaluate criterion on neighbors.
        neighbors_inds = self.get_neighbors(self.current_ind)
        neighbors_ivrs = []

        print("Evaluating criterion among {} candidates.".format(neighbors_inds.shape[0]))

        # Compute criterion for each candidate.
        for ind in neighbors_inds:
            # Look one step ahead.
            inner_ivrs = []
            inner_inds = self.get_neighbors(ind)
            for ind2 in inner_inds:
                print("Evaluating inner point {}.".format(ind2)
                # Combined observation operator for candidate location and future step.
                candidate_G = self.G[np.array([ind, ind2]),:]
                ivr = self.gp.IVR(candidate_G, self.data_std, 
                                weights=self.current_coverage)
                inner_ivrs.append(ivr)
            # Get best inner point.
            tmp_ind = np.argmax(inner_ivrs)
            best_ind = neighbors_inds[tmp_ind]
            best_ivr = inner_ivrs[best_ind]

            # Append the best inner IVR to the list of 
            # IVRs of the candidates.
            neighbors_ivrs.append(best_ivr)

        # Find best candidate.
        tmp_ind = np.argmax(neighbors_ivrs)
        best_ind = neighbors_inds[tmp_ind]
        return best_ind
