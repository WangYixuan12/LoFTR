import os
import pickle
import numpy as np
import open3d
from object_utils import rigid_transform_3D, tuple2array, points2homo
from eval_utils import recon_eval, geo_corr_eval


class Object(object):
    def __init__(self, load_dir, state_idx=15, num_pts=3000, num_mode=2, max_iter=10):

        self.load_dir = load_dir
        self.num_mode = num_mode
        self.max_iter = max_iter
        self.threshold = 10
        self.cano_obs = np.load(os.path.join(load_dir, "0.npy"))
        self.obs = np.load(os.path.join(self.load_dir, "{}.npy".format(state_idx)))
        self.num_pts = min(num_pts, len(self.cano_obs))

        with open(os.path.join(self.load_dir, "{}.pkl".format(state_idx)), 'rb') as f:
            meta = pickle.load(f)
        self.gt_trans = []
        gt_corr = np.empty(len(self.cano_obs))
        for idx, t in enumerate(meta.keys()):
            transformation = tuple2array(t)
            self.gt_trans.append(transformation)
            pc_idx = meta[t]
            gt_corr[pc_idx] = idx

        while True:
            sample_indices = np.random.choice(np.arange(len(self.cano_obs)), self.num_pts)
            self.cano_sample = self.cano_obs[sample_indices, :]
            self.obs_sample = self.obs[sample_indices, :]
            self.gt_corr = gt_corr[sample_indices]
            # brute way to ensure cover all modes in sampled points
            if self.gt_corr.max() == self.num_mode-1:
                break

    def init_group(self):
        indices = np.random.randint(self.num_mode, size=self.num_pts)
        return indices

    def compute_trans(self, indices):
        trans = []
        for mode_idx in range(self.num_mode):
            src = self.cano_sample[indices == mode_idx]
            trg = self.obs_sample[indices == mode_idx]
            transformation = rigid_transform_3D(src.T, trg.T)
            trans.append(transformation)
        trans = np.stack(trans)  # (num_mode, 3, 4)
        return trans

    def assign_group(self, trans):
        cano_h = points2homo(self.cano_sample)
        est_obs = trans.dot(cano_h.T).transpose(0, 2, 1)  # (num_mode, * ,3)
        indices = np.argmin(np.linalg.norm(est_obs - self.obs_sample[None, :, :], axis=2), axis=0)
        return indices

    def kmeans(self):
        old_indices = self.init_group()
        for iter in range(self.max_iter):
            trans = self.compute_trans(old_indices)
            new_indices = self.assign_group(trans)
            if np.sum((new_indices-old_indices)!=0) < self.threshold:
                break
            old_indices = new_indices
        return trans

    def eval(self, trans):
        est_obs, pred_corr, recon_err = recon_eval(trans, self.cano_sample, self.obs_sample)
        print("reconstruction error", recon_err)

        geo_err, corr_acc = geo_corr_eval(self.gt_corr, pred_corr, self.gt_trans, trans)
        print("geometry error", geo_err)
        print("correspondence accuracy", corr_acc)
        return est_obs, pred_corr


if __name__ == "__main__":
    indices = [20]
    # name = "laptop"
    obj_id = "10655"
    num_mode = 3

    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [[255, 255, 0]]]
    for idx in indices:
        asset = Object(load_dir=obj_id, state_idx=idx, num_mode=num_mode, num_pts=30000)

        # point_cloud = open3d.geometry.PointCloud()
        # colors = np.expand_dims(np.array([145/255, 191/255, 219/255]), axis=0).repeat(len(laptop.obs), axis=0)
        # point_cloud.points = open3d.utility.Vector3dVector(laptop.obs)
        # point_cloud.colors = open3d.utility.Vector3dVector(colors)
        # vis = open3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(point_cloud)
        # vis.run()
        # vis.capture_screen_image(os.path.join("{}_{}.jpg".format(name, idx)))
        # vis.destroy_window()

        trans = asset.kmeans()
        est_obs, pred_corr = asset.eval(trans)
        draw = []
        for mode_idx in range(asset.num_mode):
            pc = est_obs[pred_corr == mode_idx]
            pc_color = colors[mode_idx]
            pc_color = np.expand_dims(pc_color, axis=0).repeat(len(pc), axis=0)
            point_cloud = open3d.geometry.PointCloud()
            point_cloud.points = open3d.utility.Vector3dVector(pc)
            point_cloud.colors = open3d.utility.Vector3dVector(pc_color)
            draw.append(point_cloud)

        open3d.visualization.draw_geometries(draw)




