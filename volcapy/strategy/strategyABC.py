""" Sequential uncertainty reduction on excursion sets. Meta class defining a
general strategy, the only thing that varies among the different ones is how we
select the next point.

"""
from abc import ABC, abstractmethod
from scipy.spatial import KDTree
import numpy as np
import os
import torch
from volcapy.update.updatable_covariance import UpdatableGP


class StrategyABC(ABC):
    def __init__(self, updatable_gp, candidates,
            G, data_feed, lower=None, upper=None):
        """ Abstract Base Class to run sequential design strategies. The only
        thing that changes among different types of strategies is how the next
        point to visit is chosen. Hence, this class only requires the end-user
        to implement the method self.get_next_ind, which returns the index (in
        self.candidates) of the next point to visit.

        Parameters
        ----------
        updatable_gp: UpdatableGP
            Gaussian process model.
        candidates: (n_candidates, n_dims) Tensor
            Coordinates of candidate observation locations.
        G: Tensor
            Forward operator.
        data_feed: function(int)
            Function that gives observations at a given location (defined by
            index in the candidates list.
        lower: float or None
            Threshold defining the lower bound of the excursion set.
            If none, the defaults to -infinity.
        upper: float or None
            Threshold defining the upper bound of the excursion set.
            If none, the defaults to +infinity.

        """
        self.gp = updatable_gp
        self.candidates = candidates
        self.tree = KDTree(self.candidates)

        self.G = G
        self.data_feed = data_feed

        self.lower = lower
        self.upper = upper

        # Defines the neighbor structure, i.e. how many neighbors a cell is
        # supposed to have.
        self.n_neighbors = 6

    def get_nearest_neighbors(self, ind):
        """ Get the neighbors of a candidate.
        
        Parameters
        ----------
        ind: int
            Index of the node in the candidates list.

        Returns
        -------
        neighbors_inds: (N, dim) Tensor
            Indices of the neighboring points in the candidate list.

        """
        ind_tensor = ind
        if not torch.is_tensor(ind):
            ind_tensor = torch.tensor(ind)
        ind_tensor = ind_tensor.long()

        point_coord = self.candidates[ind]
        _, neighbors_inds = self.tree.query(
                point_coord, k=self.n_neighbors + 1)

        # Remove the point istelf from the list.
        neighbors_inds = neighbors_inds[neighbors_inds != int(ind)]
        neighbors_inds = torch.from_numpy(neighbors_inds).long()

        return neighbors_inds

    def get_neighbors_bigstep(self, ind, r, rmin=None):
        """ This is used for stepping a lot.
        It find all neighbors within a given radius.
        Means we can jump by more than one.
        
        Parameters
        ----------
        ind: int
            Index of the node in the candidates list.
        r: float
            Get all neighbors within that radius.
        rmin: float
            If specified, then only consider neighbors that are more than rmin
            away from the current point.

        Returns
        -------
        neighbors_inds: (N, dim) Tensor
            Indices of the neighboring points in the candidate list.

        """
        ind_tensor = ind
        if not torch.is_tensor(ind):
            ind_tensor = torch.tensor(ind)
        ind_tensor = ind_tensor.long()

        point_coord = self.candidates[ind]
        neighbors_inds = self.tree.query_ball_point(point_coord, r=r)

        # If rmin speficied, remove neighbors that are too close.
        if rmin is not None:
            close_neighbors_inds = self.tree.query_ball_point(point_coord, r=rmin)
            neighbors_inds = list(set(neighbors_inds) - set(close_neighbors_inds))

        neighbors_inds = np.array(neighbors_inds)

        # Remove the point istelf from the list.
        neighbors_inds = neighbors_inds[neighbors_inds != int(ind)]
        neighbors_inds = torch.from_numpy(neighbors_inds).long()

        return neighbors_inds

    @abstractmethod
    def get_next_ind(self):
        """ Defines where to go next (according to some criterion).
        
        Returns
        -------
        next_ind: int
            Index (in self.candidates) of the next point to visit.

        """
        pass

    def run(self, start_ind, n_steps, data_std,
            output_folder, max_step=None, min_step=None, restart_from_save=None):
        """ Run the startegy. Note that this works with any criterion to choose
        the next point, the only requirement is that selt.get_next_ind is
        defined before running the strategy.

        Parameters
        ----------
        start_ind
        n_steps: int
            Number of steps to run the strategy for.
        data_std: float
            Standard deviation of observation noise (homoscedactic).
        output_folder: string
            Path to folder where to save results.
        max_step: float
            If provided, then instead of only walking to neighbors at each
            step, can go to any cell within distance max_step.
        min_step: float
            If provided, then only consider neigbors farther away than min_step
            (must be used in conjunction with max_step).
        restart_from_save: string
            If a path to a folder is provided, then will restart the run from
            the saved data and finish it.

        """
        if restart_from_save is None:
            self.current_ind = start_ind
            self.visited_inds = []
            self.observed_data = []
            self.n_steps = n_steps
            self.data_std = data_std
            self.max_step = max_step
            self.min_step = min_step
        else:
            self.visited_inds = list(np.load(os.path.join(output_folder, "visited_inds.npy")))
            i = len(self.visited_inds) - 1
            print("Restarting from step {}.".format(i))
            self.observed_data = list(np.load(os.path.join(output_folder, "observed_data.npy"),
                    allow_pickle=True))
            self.gp = UpdatableGP.load(os.path.join(output_folder, "gp_state.pkl"))
            print(self.gp.mean.m)
    
            metadata = np.load(os.path.join(output_folder, "metadata.npy"),
                    allow_pickle='TRUE').item()
            self.current_ind = metadata['next_ind_to_visit'].item()
            print(self.current_ind)
            self.max_step = metadata['max_step']
            try:
                self.min_step = metadata['min_step']
            except:
                pass

            self.data_std = metadata['data_std']

            # Remaining steps to perform.
            self.n_steps = metadata['remaining_steps']

        # Change the get neighbors routine if can jump more that one step.
        if self.max_step is not None:
            self.get_neighbors = lambda x: self.get_neighbors_bigstep(x,
                    r=self.max_step, rmin=self.min_step)
        else: self.get_neighbors = lambda x: self.get_nearest_neighbors(x)

        for i in range(n_steps):
            # Observe at currennt location and update model.
            self.visited_inds.append(self.current_ind)
            y = self.data_feed(self.current_ind).detach().float()
            self.observed_data.append(y.detach().float().numpy())
            G = self.G[self.current_ind,:].reshape(1, -1)
            self.gp.update(G, y, data_std)

            # Extract current coverage function (after observing at current
            # location).
            self.current_coverage = self.gp.coverage(self.lower, self.upper)

            # Now evaluate where to go next.
            next_ind = self.get_next_ind()
            self.current_ind = next_ind
            print("Go to cell {} for step {}.".format(self.current_ind,
                    len(self.visited_inds)))

            # Save coverage each iteration.
            self.save_state(output_folder, coverage_only=True)

            # Save full state every 5 iterations.
            if i % 5 == 0:
                self.save_state(output_folder)

        return visited_inds, observed_data

    def save_state(self, output_folder, coverage_only=False):
        """ Save the current state of the run, so can be re-launched in case of
        interrupt.

        Parameters
        ----------
        output_folder: string
            Path to folder where to save.
        coverage_only: bool, defaults to False
            If True, then only save the current coverage.

        """
        i = len(self.visited_inds) - 1
        np.save(os.path.join(output_folder, "coverage_{}.npy".format(i)),
                    self.current_coverage)

        if coverage_only is False:
            np.save(os.path.join(output_folder, "visited_inds.npy"), self.visited_inds)
            np.save(os.path.join(output_folder, "observed_data.npy"), self.observed_data)
            self.gp.save(os.path.join(output_folder, "gp_state.pkl"))
    
            metadata = {'max_step': self.max_step, 'min_step': self.min_step,
                    'next_ind_to_visit': self.current_ind,
                    'data_std': self.data_std, 'i': i, 'remaining_steps': self.n_steps - i}
            np.save(os.path.join(output_folder, "metadata.npy"), metadata)

        """
            # Save current mean if needed.
            post_mean = self.gp.mean_vec.detach().numpy()
            # TODO: Fix and include upper.
            plugin_est_inds, _ = np.where(
                    (post_mean >= self.lower))
            print("Excursion set with {} elements".format(plugin_est_inds.shape[0]))
            np.save(
                os.path.join(
                        output_folder,
                        "plugin_est_inds_{}.npy".format(i)),
                plugin_est_inds)
        """

    def save_plugin_estimate(self, visited_inds, data_std, output_folder):
        """ Given a list of indices to visit, compute and save posterior means.

        """
        for i, current_ind in enumerate(visited_inds):
            # Observe and update model.
            current_ind = int(current_ind)

            y = self.data_feed(current_ind)
            G = self.G[current_ind,:].reshape(1, -1)
            self.gp.update(G, y, data_std)

            post_mean = self.gp.mean_vec.detach().numpy()
            """
            plugin_est_inds = np.where(
                    (post_mean >= self.lower) & (post_mean <= self.upper))
            """
            # TODO: Fix and include upper.
            plugin_est_inds, _ = np.where(
                    (post_mean >= self.lower))
            print("Excursion set with {} elements".format(plugin_est_inds.shape[0]))
            np.save(
                os.path.join(
                        output_folder,
                        "plugin_est_inds_{}.npy".format(i)),
                plugin_est_inds)
