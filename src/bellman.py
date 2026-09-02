"""
SurvTD 핵심 수학 엔진: Survival Bellman Operator, λ-Return 및 Cramér Loss
"""
import torch
import torch.nn as nn

def compute_survival_bellman_target(p_next, S_curr, dt_steps, event, K=30):
    """
    1-Step Survival Bellman Operator:
    T p_j(s) = I[Event in dt] + S_j(dt) * p_{j+1}(s - dt)
    """
    target_pmf = torch.zeros(K, device=p_next.device)
    dt_int = max(1, min(int(dt_steps), K - 1))
    
    if event > 0.5:
        target_pmf[min(dt_int - 1, K - 1)] = 1.0
        return target_pmf
        
    S_dt = S_curr[min(dt_int - 1, K - 1)]
    if dt_int < K:
        shifted_p_next = p_next[:K - dt_int] * S_dt
        target_pmf[dt_int:] = shifted_p_next.detach()
    return target_pmf

def compute_lambda_returns(target_pmfs, survivals, dt_list, is_failure, lam=0.6, K=30, dt_res=0.05):
    """
    SurvTD(λ) λ-Return 후진 재귀 (Backward View Recursion):
    G^\lambda_j(s) = I_j + S_j(\Delta t) * [ (1 - \lambda) * p_{j+1}(s - \Delta t) + \lambda * G^\lambda_{j+1}(s - \Delta t) ]
    """
    seq_len = target_pmfs.shape[0]
    G_lambda = [None] * seq_len
    G_lambda[-1] = target_pmfs[-1].clone().detach()
    
    for t in range(seq_len - 2, -1, -1):
        dt = dt_list[t]
        dt_steps = max(1, min(int(round(dt / dt_res)), K - 1))
        
        is_terminal = (t == seq_len - 2) and is_failure
        target_p = torch.zeros(K, device=target_pmfs.device)
        
        if is_terminal:
            target_p[min(dt_steps - 1, K - 1)] = 1.0
            G_lambda[t] = target_p
        else:
            S_dt = survivals[t, min(dt_steps - 1, K - 1)]
            
            p_next_shifted = torch.zeros(K, device=target_pmfs.device)
            if dt_steps < K:
                p_next_shifted[dt_steps:] = target_pmfs[t + 1, :K - dt_steps] * S_dt
                
            G_next_shifted = torch.zeros(K, device=target_pmfs.device)
            if dt_steps < K:
                G_next_shifted[dt_steps:] = G_lambda[t + 1][:K - dt_steps] * S_dt
                
            G_lambda[t] = ((1.0 - lam) * p_next_shifted + lam * G_next_shifted).detach()
            
    return G_lambda

def cramer_distance_loss(pred_cdf, target_cdf):
    """
    Cramér Distance Loss (L2 distance on cumulative risk distributions):
    L = 1/K * \sum_{s=1}^K (F_pred(s) - F_target(s))^2
    """
    return torch.mean((pred_cdf - target_cdf) ** 2)
