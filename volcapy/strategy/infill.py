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

    def run(self, data_std, output_folder, n_data_splits=None):
        """ Re-implement the run method so we can benefit from all the
        automation implemented in the startegyABC class, in particular the
        saving.

        Parameters
        ----------
        data_std: float
            Standard deviation of observation noise (homoscedactic).
        output_folder: string
            Path to folder where to save results.
        n_data_splits: int or None
            In how many parts we should split the whole data for absorption.
            Note that this is necessary, since the covariance pushforward for
            the whole dataset might be too large for memory.

        """
        # Those need to be set to arbitrary values for compatibility with
        # strategyABC.
        self.max_step = 0
        self.current_ind = 0
        self.n_steps = 0
        self.data_std = data_std

        self.visited_inds = []
        self.observed_data = []

        print(np.random.shuffle(list(range(self.candidates.shape[0]))))

        # Split indices in groups or not.
        # Start by shuffing in place.
        inds_to_iter = list(range(self.candidates.shape[0]))
        np.random.shuffle(inds_to_iter)

        if n_data_splits is not None:
            inds_to_iter = np.array_split(
                    inds_to_iter,
                    n_data_splits)

        for i, sub_inds in enumerate(inds_to_iter):
            print("Processing split {} / {}.".format(i, n_data_splits))
            print("Split contains surface inds {}.".format(sub_inds))

            self.visited_inds.append(sub_inds)
            y = self.data_feed(sub_inds)
            self.observed_data.append(y)
            
            # Find the shape of the data we are absorbing.
            if isinstance(sub_inds, int):
                data_shape = 1
            else:
                data_shape = sub_inds.shape[0]
            G = self.G[sub_inds,:].reshape(data_shape, -1)
            self.gp.update(G, y, data_std)
            
            # Save full state every few iterations.
            if i % 10 == 0:
                self.current_coverage = self.gp.coverage(self.lower, self.upper)
                self.save_state(output_folder)

        # Extract infill coverage function.
        self.current_coverage = self.gp.coverage(self.lower, self.upper)

        # Save full state.
        self.save_state(output_folder)

        return self.visited_inds, self.observed_data
