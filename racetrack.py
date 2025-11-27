import numpy as np

import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.axes as axes

class RaceTrack:

    def __init__(self, centerline_path: str, raceline_path: str):
        centerline_data = np.loadtxt(centerline_path, comments="#", delimiter=",")
        self.centerline = centerline_data[:, 0:2]
        self.centerline = np.vstack((self.centerline[-1], self.centerline, self.centerline[0]))

        self.raceline = np.loadtxt(raceline_path, comments="#", delimiter=",")
        self.raceline = np.vstack((self.raceline[-1], self.raceline, self.raceline[0]))
        self.raceline = self._align_paths()

        centerline_gradient = np.gradient(self.centerline, axis=0)
        # Unfortunate Warning Print: https://github.com/numpy/numpy/issues/26620
        centerline_cross = np.cross(centerline_gradient, np.array([0.0, 0.0, 1.0]))
        centerline_norm = centerline_cross*\
            np.divide(1.0, np.linalg.norm(centerline_cross, axis=1))[:, None]

        centerline_norm = np.delete(centerline_norm, 0, axis=0)
        centerline_norm = np.delete(centerline_norm, -1, axis=0)

        self.centerline = np.delete(self.centerline, 0, axis=0)
        self.centerline = np.delete(self.centerline, -1, axis=0)
        self.raceline = np.delete(self.raceline, 0, axis=0)
        self.raceline = np.delete(self.raceline, -1, axis=0)

        # Compute track left and right boundaries
        self.right_boundary = self.centerline[:, :2] + centerline_norm[:, :2] * np.expand_dims(centerline_data[:, 2], axis=1)
        self.left_boundary = self.centerline[:, :2] - centerline_norm[:, :2]*np.expand_dims(centerline_data[:, 3], axis=1)

        blend = 0.7 # 0 -> centerline, 1 -> raceline
        self.target_path = (1 - blend) * self.centerline + blend * self.raceline

        # Compute initial position and heading
        self.initial_state = np.array([
            self.target_path[0, 0],
            self.target_path[0, 1],
            0.0, 0.0,
            np.arctan2(
                self.centerline[1, 1] - self.centerline[0, 1], 
                self.centerline[1, 0] - self.centerline[0, 0]
            )
        ])

        # Matplotlib Plots
        self.code = np.empty(self.centerline.shape[0], dtype=np.uint8)
        self.code.fill(path.Path.LINETO)
        self.code[0] = path.Path.MOVETO
        self.code[-1] = path.Path.CLOSEPOLY

        self.mpl_centerline = path.Path(self.centerline, self.code)
        self.mpl_right_track_limit = path.Path(self.right_boundary, self.code)
        self.mpl_left_track_limit = path.Path(self.left_boundary, self.code)

        self.mpl_centerline_patch = patches.PathPatch(self.mpl_centerline, linestyle="-", fill=False, lw=0.3)
        self.mpl_right_track_limit_patch = patches.PathPatch(self.mpl_right_track_limit, linestyle="--", fill=False, lw=0.2)
        self.mpl_left_track_limit_patch = patches.PathPatch(self.mpl_left_track_limit, linestyle="--", fill=False, lw=0.2)

        self.code = np.empty(self.raceline.shape[0], dtype=np.uint8)
        self.code.fill(path.Path.LINETO)
        self.code[0] = path.Path.MOVETO
        self.code[-1] = path.Path.CLOSEPOLY

        self.mpl_raceline = path.Path(self.raceline, self.code)
        self.mpl_raceline_patch = patches.PathPatch(self.mpl_raceline, linestyle="--", fill=False, lw=0.3, color="red")

    def _align_paths(self):
        def get_normalized_progress(path):
            diffs = np.diff(path, axis=0, prepend=path[0].reshape(1, -1))
            dists = np.linalg.norm(diffs, axis=1)
            
            cumulative_dist = np.cumsum(dists)
            
            if cumulative_dist[-1] == 0:
                return np.zeros_like(cumulative_dist)
            return cumulative_dist / cumulative_dist[-1]

        center_s = get_normalized_progress(self.centerline)
        race_s = get_normalized_progress(self.raceline)

        new_x = np.interp(center_s, race_s, self.raceline[:, 0])
        new_y = np.interp(center_s, race_s, self.raceline[:, 1])
        aligned_raceline = np.column_stack((new_x, new_y))

        return aligned_raceline

    def plot_track(self, axis : axes.Axes):
        axis.add_patch(self.mpl_centerline_patch)
        axis.add_patch(self.mpl_right_track_limit_patch)
        axis.add_patch(self.mpl_left_track_limit_patch)
        axis.add_patch(self.mpl_raceline_patch)
