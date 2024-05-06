import os
import time
import torch
import psutil
from torch import nn
import numpy as np
import torch.nn.functional as F

from utils import torch_kl


def hierarchical_contrastive_loss(
        h1, h2, 
        tau1, tau2, 
        tembed_jsd_task_head,
        tembed_pred_task_head,
        weights, 
        temporal_unit=0,
    ):
    """
        Applies the losses of all pretext tasks hierarchically,
        at different time-scales, with the given task weightings.

    Args:
        h1 (_type_): Time-series representations under context 1.
        h2 (_type_): Time-series representations under context 2.
        tau1 (_type_): Time-embeddings for the given window.
        tau2 (_type_): Time-embeddings for the given window.
        tembed_jsd_task_head (_type_): Model to use for the time-embedding JSD prediction task.
        tembed_pred_task_head (_type_): Model to use for the time-embedding-conditioned forecasting task.
        weights (_type_): Pretext task weights.
        temporal_unit (int, optional): Smallest unit of time to apply temporal tasks. Defaults to 0.

    Returns:
        The loss aggregated over all pretext tasks and time scales.
    """
    loss = torch.tensor(0., device=h1.device)
    d = 0
    while h1.size(1) > 1:
        if weights['instance_contrast'] != 0:
            loss += weights['instance_contrast'] * instance_contrastive_loss(h1, h2)
        if d >= temporal_unit:
            if weights['tembed_jsd_pred'] != 0:
                jsd_loss = tembed_jsd_pred_loss(h1, h2, tau1, tau2, tembed_jsd_task_head)
                loss += weights['tembed_jsd_pred'] * jsd_loss
            if weights['temporal_contrast'] != 0:
                t_loss = temporal_contrastive_loss(h1, h2)
                loss += weights['temporal_contrast'] * t_loss 
            if weights['tembed_cond_pred'] != 0:
                pred_loss = tembed_cond_pred_loss(h1, h2, tau1, tau2, tembed_pred_task_head) 
                loss += weights['tembed_cond_pred'] * pred_loss

        d += 1
        h1 = F.max_pool1d(h1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        h2 = F.max_pool1d(h2.transpose(1, 2), kernel_size=2).transpose(1, 2)

        if tau1 is not None and tau2 is not None:
            tau1 = F.max_pool1d(tau1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            tau2 = F.max_pool1d(tau2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if h1.size(1) == 1:
        if weights['instance_contrast'] != 0:
            loss += weights['instance_contrast'] * instance_contrastive_loss(h1, h2)
        d += 1
    return loss / d

def instance_contrastive_loss(h1, h2):
    """
        Instance Contrastive task, see TS2Vec paper for details.

    Args:
        h1 (_type_): Time-series representations under context 1.
        h2 (_type_): Time-series representations under context 2.

    Returns:
        The computed contrastive loss
    """
    B, T = h1.size(0), h1.size(1)
    if B == 1:
        return h1.new_tensor(0.)
    z = torch.cat([h1, h2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]     # T x 2B x (2B-1)
    logits = -F.log_softmax(logits, dim=-1)             # T x 2B x (2B-1)
    
    i = torch.arange(B, device=h1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(h1, h2):
    """
        Temporal Contrastive task, see TS2Vec paper for details.

    Args:
        h1 (_type_): Time-series representations under context 1.
        h2 (_type_): Time-series representations under context 2.

    Returns:
        The computed contrastive loss
    """
    ts = time.time()
    B, T = h1.size(0), h1.size(1)
    if T == 1:
        return h1.new_tensor(0.)
    z = torch.cat([h1, h2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=h1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def tembed_jsd_pred_loss(h1, h2, tau1, tau2, tembed_jsd_task_head):
    """
        Time-embedding JSD prediction task. See T-Rep paper for details.

    Args:
        h1 (_type_): Time-series representations under context 1.
        h2 (_type_): Time-series representations under context 2.
        tau1 (_type_): Time-embeddings for the given window.
        tau2 (_type_): Time-embeddings for the given window.
        tembed_jsd_task_head (_type_): Model to use for the predictions. 

    Returns:
       The computed MSE loss between predicted and actual JSD.
    """
    B, T, C_tau = tau1.shape
    C_h = h1.shape[2]

    pair_indices = get_tembed_jsd_pairs(
        num_taus=B*T if B*T % 2 == 0 else B*T - 1,
        n_pairs=800, 
        device=h1.device
    )

    tau_pairs = tau1 \
                .reshape(-1, C_tau)[pair_indices.flatten()] \
                .reshape(pair_indices.shape[0], 2, C_tau)

    repr1_pairs = h1 \
                .reshape(-1, C_h)[pair_indices.flatten()] \
                .reshape(pair_indices.shape[0], 2, C_h)
    repr2_pairs = h2 \
                .reshape(-1, C_h)[pair_indices.flatten()] \
                .reshape(pair_indices.shape[0], 2, C_h)
    
    M = (tau_pairs[:, 0, :] + tau_pairs[:, 1, :]) / 2
    tau_kl1 = torch_kl(tau_pairs[:, 0, :], M).sum(dim=1)[..., None]
    tau_kl2 = torch_kl(tau_pairs[:, 1, :], M).sum(dim=1)[..., None]
    js_div = 0.5 * tau_kl1 + 0.5 * tau_kl2

    repr1_pair_diffs = repr1_pairs[:, 0, :] - repr2_pairs[:, 1, :]
    pred_js_div = tembed_jsd_task_head(repr1_pair_diffs)
    loss = nn.MSELoss()
    return loss(pred_js_div, js_div)


def get_tembed_jsd_pairs(num_taus, n_pairs=800, device='cpu'):
    """
        Randomly pick pairs to use for the time-embedding divergence
        prediction task.

    Args:
        num_taus (_type_): Number of data points to pick from.
        n_pairs (int, optional): Numbr of pairs to produce. Defaults to 800.
        device (str, optional): Device to store data on. Defaults to 'cpu'.

    Returns:
        Randomly picked pairs for the time-embedding divergence
        prediction task. 
    """
    half_pairs = min(n_pairs, num_taus) // 2
    diff_pairs = torch.randperm(num_taus).reshape(-1, 2)[:half_pairs].to(device)
    identity_pairs = torch.randperm(num_taus).repeat_interleave(2).reshape(-1, 2)[:half_pairs].to(device)
    return torch.cat((identity_pairs, diff_pairs), axis=0).long()


def tembed_cond_pred_loss(h1, h2, tau1, tau2, task_head, max_time_dist=10):
    """
        Time-embedding-conditioned prediction task. Pairs of (input, target)
        predictions are picked at random, a prediction is made given the input
        representation and the time-embedding at the target's timestep, and an MSE
        loss is computed between the target and the prediction.

    Args:
        h1 (_type_): Time-series representations in context 1.
        h2 (_type_): Time-series representations in context 2.
        tau1 (_type_): Time-embeddings for the given time window.
        tau2 (_type_): Time-embeddings for the given time window.
        task_head (_type_): Model used for the prediction task
        max_time_dist (int, optional): Maximum forecasting range. Defaults to 10.

    Returns:
        The computed MSE forecasting loss over the current batch.
    """
    B, T, C = h1.shape
    C_tau = tau1.shape[-1]
    pairs, context = get_tembed_pred_pairs(
        max_range=T,
        max_dist=min(max_time_dist, T),
        B=B,
        n_pairs=800,
        device=h1.device,
    )

    h = torch.stack((h1, h2), axis=1)
    tau = torch.stack((tau1, tau2), axis=1)
    chosen_reprs = h[:, :, pairs][torch.arange(B).long(), context.long()].reshape(-1, 2, C)
    chosen_taus = tau[:, :, pairs][torch.arange(B).long(), context.long()][:, :, 1, :].reshape(-1, C_tau)
    
    X = torch.cat((chosen_reprs[:, 0, :], chosen_taus), axis=-1)
    y =  chosen_reprs[:, 1, :]

    pred = task_head(X)
    loss = nn.MSELoss()
    return loss(pred, y)
    

def get_tembed_pred_pairs(max_range, max_dist, B, n_pairs, device='cpu'):
    """
        Generates pairs of (input, target) for the time-embedding conditioned
        forecasting task.

    Args:
        max_range (_type_): End of window in which to sample pairs.
        max_dist (_type_): Maximum forecasting horizon (used both forwards and backwards).
        B (_type_): Batch size
        n_pairs (_type_): Number of (input, target) pairs to return
        device (str, optional): Device to send data to. Defaults to 'cpu', otherwise 'cuda:'.

    Returns:
        pairs: The indices at which to sample (input, target) pairs.
        context: The context to choose the input or target from, either h1 or h2.
    """
    pairs_per_batch = n_pairs // B
    
    context = torch.randint(0, 2, size=(B,)).long()

    if max_range > max_dist + 1:
        basis = torch.randint(0, max_range - max_dist, size=(pairs_per_batch,)).to(device)

        # Choose deltas backwards and forwards in time
        deltas = torch.randint(- max_dist + 1, max_dist, size=(pairs_per_batch,)).to(device)
        pairs = torch.stack([basis, basis + deltas], axis=1).long()
    else:
        # When time window is shorter than max distance, grab random permutations in window
        pairs = torch.randint(max_range, size=(pairs_per_batch * 2,)).reshape(pairs_per_batch, 2).to(device)

    return pairs, context
