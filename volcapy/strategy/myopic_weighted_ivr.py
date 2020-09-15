""" Sequential uncertainty reduction on excursion sets using weighted IVR
criterion and myopic path planing.

"""
from scipy.spatial import KDTree
import numpy as np
import os
import torch


class MyopicStrategy:
    def __init__(self, updatable_gp, candidates,
            G, data_feed, lower=None, upper=None):
        """

        Parameters
        ----------
        updatable_gp: UpdatableGP
            Gaussian process model.
        candidates: (n_candidates, n_dims) Tensor
            Coordinates of candidate observation locations.
        data_feed: function(int)
            Function that gives observations at a given location (defined by
            index in the candidates list.
        lower
        upper

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

    def get_neighbors(self, ind):
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

    def run(self, start_ind, n_steps, data_std):
        current_ind = start_ind
        visited_inds = []
        observed_data = []
        ivrs = []

        for i in range(n_steps):
            # Observe and update model.
            y = self.data_feed(current_ind)
            G = self.G[current_ind,:].reshape(1, -1)
            self.gp.update(G, y, data_std)

            observed_data.append(y)
            visited_inds.append(current_ind)

            # Evaluate criterion on neighbors.
            neighbors_inds = self.get_neighbors(current_ind)
            neighbors_ivrs = []
            for ind in neighbors_inds:
                # Observation operator for candidate location.
                candidate_G = self.G[ind,:].reshape(1, -1)

                neighbors_ivrs.append(
                        self.gp.weighted_IVR(candidate_G, data_std,
                                self.lower, self.upper))
            # Go to best neighbor.
            tmp_ind = np.argmax(neighbors_ivrs)
            best_ind = neighbors_inds[tmp_ind]
            ivrs.append(neighbors_ivrs[tmp_ind])
            current_ind = best_ind
            print("IVRS at current stage: {}.".format(neighbors_ivrs))
            print("Go to cell {}.".format(current_ind))

            np.save("visited_inds.npy", visited_inds)
            np.save("observed_data.npy", observed_data)
            np.save("ivrs.npy", ivrs)

        return visited_inds, observed_data, ivrs

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
