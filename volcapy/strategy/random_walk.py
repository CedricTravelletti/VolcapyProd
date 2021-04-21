""" Sequential uncertainty reduction on excursion sets. Class defining the
random walk acquisition function.

"""
from volcapy.strategy.strategyABC import StrategyABC
import numpy as np


class RandomWalkStrategy(StrategyABC):
    def get_next_ind(self):
        # Get list of candidates.
        neighbors_inds = self.get_neighbors(self.current_ind)

        # Choose one candidate at random.
        return np.random.choice(neighbors_inds)
