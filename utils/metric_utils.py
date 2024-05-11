import numpy as np
import torch
def top_k_acc(y_true_seq, y_pred_seq, k):
    hit = 0
    # Convert to binary relevance (nonzero is relevant).
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        top_k_rec = y_pred.argsort()[-k:][::-1]
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            hit += 1
    return hit / len(y_true_seq)


def mAP_metric(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    return rlt / len(y_true_seq)


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):

    
    if y_true_seq.shape[0] == 0:
        return 0.0
    
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    
    top_k_preds = torch.topk(y_pred, k, dim=-1).indices
    match = (top_k_preds == y_true.unsqueeze(dim=-1)).any(dim=-1)
    acc = match.sum().item() / match.size(0)
    
    return acc


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    
    if y_true_seq.shape[0] == 0:
        return 0.0
    
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    
    top_k_preds = torch.topk(y_pred, k, dim=-1).indices
    match = (top_k_preds == y_true.unsqueeze(dim=-1)).float()
    
    precisions = torch.cumsum(match, dim=-1) / torch.arange(1, k+1, device=match.device).float()
    avg_precision = torch.sum(precisions * match, dim=-1) / match.sum(dim=-1)
    avg_precision[torch.isnan(avg_precision)] = 0.0
    
    return avg_precision.mean().item()


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):

    if y_true_seq.shape[0] == 0:
        return 0.0
    
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    
    rank = torch.nonzero(torch.argsort(y_pred, descending=True) == y_true).squeeze().item() + 1
    mrr = 1.0 / rank
    
    return mrr