import os
import numpy as np
import torch
from volcapy.uq.set_estimation import vorobev_expectation_inds
from volcapy.prepare_DP_experiments import prepare_experiment_data


def prepare_helper_functions(local=False):
    # Prepare experiment.
    sample_nr = 1
    (trained_gp_model, ground_truth, volcano_grid,
         data_coords, G, data_feed, niklas_data_inds_insurf,
         accessibility_graph, base_station_node,
         THRESHOLD_small, THRESHOLD_big, _) = prepare_experiment_data(sample_nr, local)
    
    # Define the (blind) critetion to optimize.
    lower_thresold = torch.Tensor([2600.0])
    if local is False:
        lower_thresold = lower_thresold.to('cuda')
    data_std = 0.1 
    
    G_full_surface = torch.from_numpy(G)
    
    def compute_blind_wIVR(design, belief):
        belief.rewind(0)
        # Get the observation operator corresponding to the design.                     
        G = G_full_surface[np.array(design), :]                                                                                                                
        current_coverage = belief.coverage(lower_thresold, None)                        
        variance_reduction, _ = belief.covariance.compute_VR(G, data_std)
        belief.rewind(0)
        wIVR = torch.sum(variance_reduction * current_coverage)
        IVR = torch.sum(variance_reduction)
    
        return {'reward': wIVR.item(), 'reward_metadata': {'IVR': IVR.item()}}
                                                                                      
    def compute_mismatch(ground_truth, estimated_excursion_inds):  
              true_excursion_inds = (ground_truth >= lower_thresold.cpu()).nonzero()[:, 0]       
                                                                                      
              # Compute mismatch.                                                     
              mismatch = np.zeros(ground_truth.shape[0])                            
              mismatch[estimated_excursion_inds] = 1                                        
              tmp = np.zeros(ground_truth.shape[0])                                 
              tmp[true_excursion_inds] = 2                                            
              mismatch = mismatch + tmp                                               
                                                                                      
              excu_size = true_excursion_inds.shape[0] + 1                            
              p_false_pos = 100 * np.sum(mismatch == 1) / excu_size                   
              p_false_neg = 100 * np.sum(mismatch == 2) / excu_size                   
              p_correct = 100 * np.sum(mismatch == 3) / excu_size                     
                                                                                      
              return (p_false_pos, p_false_neg, p_correct)          
    
    def compute_accuracy_metrics(ground_truth, design, belief, already_updated=False):
        if not torch.is_tensor(ground_truth): ground_truth = torch.from_numpy(ground_truth)
        if already_updated is False:
            # Get the data.
            # Get the observation operator corresponding to the design.                     
            G = G_full_surface[np.array(design), :]
            y = data_feed(np.array(design))
        
            # Add the data.
            belief.update(G, y, data_std)
        
        # Compute the new variance and coverage.
        variance_new = belief.covariance.extract_variance().cpu()
        coverage_new = belief.coverage(lower_thresold, None).cpu()
        
        # Residual weighted integrated variance.
        wIV_new = torch.sum(variance_new * coverage_new)
        IBV_final = torch.sum(coverage_new * (1 - coverage_new))
        
        # Error with plug-in estimate.
        post_mean = belief.mean_vec.cpu()
        vorobev_excursion_inds = vorobev_expectation_inds(coverage_new)
        p_false_pos, p_false_neg, p_correct = compute_mismatch(ground_truth, vorobev_excursion_inds)
    
        if already_updated is False:
            # Return GP to original state.
            belief.rewind(0)
        return {'wIV_final': wIV_new.item(),
                'IBV_final': IBV_final.item(), 'p_false_pos_final': p_false_pos,
                'p_false_neg_final': p_false_neg, 'p_correct_final': p_correct}

    return compute_blind_wIVR, compute_mismatch, compute_accuracy_metrics
