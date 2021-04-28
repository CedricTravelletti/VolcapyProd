""" Sequential uncertainty reduction on excursion sets. Class defining the
infill strategy. This is not a strategy per se, since it is not sequential.
Infill ingests all surface data at once.

"""
from volcapy.strategy.strategyABC import StrategyABC
import numpy as np


class InfillStrategy(StrategyABC):
    def get_next_ind(self):
        # Need to be implemented since it is an abstact method.
        pass

    def run(self, data_std, output_folder):
        """ Re-implement the run method so we can benefit from all the
        automation implemented in the startegyABC class, in particular the
        saving.

        Parameters
        ----------
        data_std: float
            Standard deviation of observation noise (homoscedactic).
        output_folder: string
            Path to folder where to save results.

        """
        # Those need to be set to arbitrary values for compatibility with
        # strategyABC.
        self.max_step = 0
        self.current_ind = 0
        self.n_steps = 0

        self.visited_inds.append(range(self.candidates.shape[0]))
        y = self.data_feed(self.visited_inds)

        self.observed_data.append(y)
        self.gp.update(self.G, y, data_std)

        # Extract infill covewrage function.
        self.current_coverage = self.gp.coverage(self.lower, self.upper)

        # Save full state.
        self.save_state(output_folder)

        return visited_inds, observed_data
