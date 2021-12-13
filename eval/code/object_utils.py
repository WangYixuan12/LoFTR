import numpy as np


def points2homo(points):
    points = np.concatenate((points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1)
    return points


def safe_arccos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))


def SE3_distance(pose1, pose2):
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    R = np.dot(R1.T, R2)
    rot_loss = safe_arccos((R.trace() - 1) / 2)
    Tdiff = pose1[:3, 3] - pose2[:3, 3]
    tr_loss = np.linalg.norm(Tdiff)
    return rot_loss + tr_loss


def nms(cost, T, neighbor_size=0.5, threshold=1000):
    # nms
    import scipy.ndimage.filters as filters
    sel_indices = cost > threshold
    sel_ori_indices = np.arange(len(cost))[sel_indices]
    sel_cost = cost[sel_indices]
    sel_T = T[sel_indices]

    ixs = sel_cost.argsort()[::-1]
    pick_ixs = []
    while len(ixs) > 0:
        pick_ixs.append(ixs[0])
        pick_T = sel_T[ixs[0]]
        dist = np.array([SE3_distance(pick_T, rest_T) for rest_T in sel_T[ixs[1:]]])
        remove_ixs = np.where(dist < neighbor_size)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    indices = sel_ori_indices[pick_ixs]
    return indices


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    T = np.concatenate((R, t), axis=1)  # (3, 4)
    return T


def tuple2array(tuple):
    return np.asarray(tuple)
