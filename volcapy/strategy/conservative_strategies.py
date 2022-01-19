""" SUR strategies based on conservative estimates. 
This module implements the criterions defined in D. Azzimonti's 
thesis, chapter 5.2.

"""
import torch
import numpy as np
from scipy.stats import norm
from torch.distributions import Normal
from volcapy.strategy.strategyABC import StrategyABC
from volcapy.uq.set_estimation import vorobev_quantile_inds


class ConservativeStrategy(StrategyABC):
    def compute_helper_quantities(self, G, data_std):
        """ Compute helper quantities a_n, b_n defined 
        in eq. 5.9.
        WARNING: Only works for excursion above threshold.

        Parameters
        ----------
        G: Tensor (n_obs, n_cells)
            Observation operator.
        data_std: float
            Observational noise standard deviation.

        Returns
        -------
        a: Tensor (n_cells)
        b: Tensor (n_obs, n_cells)
        gamma: Tensor (n_cells)

        """
        mean = self.gp.mean_vec
        fantasy_variance, G_dash = self.gp.covariance.compute_fantasy_variance(G, data_std)
        fantasy_std = torch.sqrt(fantasy_variance).reshape(
                fantasy_variance.shape[0], 1)

        a = torch.div(
                mean - self.lower * torch.ones(mean.shape),
                fantasy_std)
        K_q = G.double() @ G_dash.double()
        b = torch.div(G_dash.double() @ torch.inverse(K_q),
                fantasy_std.repeat(1, K_q.shape[0]).double()).t().float()

        gamma = torch.sqrt(b.t().double() @ K_q.double() @ b.double())
        return (a, b, gamma.float())

    def compute_J_meas(self, G, data_std, rho):
        """ Compute the J^{MEAS} criterions (Eq.5.13) for a given candidate 
        location.

        Parameters
        ----------
        G: Tensor (n_obs, n_cells)
            Observation operator.
        data_std: float
            Observational noise standard deviation.
        rho: float
            Conservative level for the criterion.

        Returns
        -------
        criterion: float

        """
        a, b, gamma = compute_helper_quantities(self, G, data_std)
        J_meas = Normal(loc=0, std=1).cdf(
                torch.div(
                    a - norm.ppf(rho) * torch.ones(a.shape),
                    gamma)).sum()
        return J_meas

    def get_next_ind(self):
        # TODO: temporary.
        # Compute inclusion proba for vorobev quantile at level rho.
        rho = 0.95
        vorb_quantile_inds = vorobev_quantile_inds(self.current_coverage, rho)
        inclusion_proba = self.compute_inclusion_probability(vorb_quantile_inds)
        print("Inclusion proba: {}".format(inclusion_proba))

        # Evaluate criterion on neighbors.
        neighbors_inds = self.get_neighbors(self.current_ind)
        neighbors_ivrs = []

        print("Evaluating criterion among {} candidates.".format(neighbors_inds.shape[0]))

        # Compute criterion for each candidate.
        for ind in neighbors_inds:
            # Observation operator for candidate location.
            candidate_G = self.G[ind,:].reshape(1, -1)
            ivr = self.gp.IVR(candidate_G, self.data_std, 
                                weights=self.current_coverage)
            neighbors_ivrs.append(ivr)

        # Find best candidate.
        tmp_ind = np.argmax(neighbors_ivrs)
        best_ind = neighbors_inds[tmp_ind]
        return best_ind
