import numpy as np
from eval.object_utils import SE3_distance, points2homo


def recon_eval(trans, cano, obs):
    est_trans = np.stack(trans)
    cano_h = points2homo(cano)
    est_obs = est_trans.dot(cano_h.T).transpose(0, 2, 1)  # (num_mode, *, 3)
    dist = np.linalg.norm(est_obs - obs[None, :, :], axis=2)  # (num_mode, *)
    indices = np.argmin(dist, axis=0)  # (*)

    est_obs = est_obs.transpose(1, 2, 0)  # (*, 3, num_mode)
    group_id = np.expand_dims(indices, axis=1).repeat(est_obs.shape[1], axis=1)  # (*, 3)
    row, column = np.meshgrid(np.arange(est_obs.shape[0]), np.arange(est_obs.shape[1]), indexing='ij')
    est_obs = est_obs[row, column, group_id]
    err = np.linalg.norm(est_obs - obs, axis=1).mean(axis=0)
    return est_obs, indices, err


def geo_eval(gt_trans, pred_trans):
    # row gt trans, column pred trans
    from scipy.optimize import linear_sum_assignment
    cost = np.empty((len(gt_trans), len(pred_trans)))
    for i, gt in enumerate(gt_trans):
        for j, pred in enumerate(pred_trans):
            geo_err = SE3_distance(gt, pred)
            cost[i, j] = geo_err
    row_ind, col_ind = linear_sum_assignment(cost)
    geo_err = cost[row_ind, col_ind].sum()
    return geo_err, col_ind


def corr_eval(gt_corr, pred_corr):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt_corr, pred_corr)
    return accuracy


def geo_corr_eval(gt_corr, pred_corr, gt_trans, pred_trans):
    geo_err, col_ind = geo_eval(gt_trans, pred_trans)
    gt_map_corr = np.empty_like(gt_corr)
    print(col_ind)
    for idx, col_idx in enumerate(col_ind):
        gt_map_corr[gt_corr == idx] = col_idx
    corr_accuracy = corr_eval(gt_map_corr, pred_corr)

    return geo_err, corr_accuracy
