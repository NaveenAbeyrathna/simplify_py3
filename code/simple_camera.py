import chumpy as ch
import numpy as np

class SimpleProjectPoints(ch.Ch):
    dterms = 'f', 'rt', 't', 'c', 'v'

    def compute_r(self):
        # f: (2,) focal length
        # rt: (3,) axis-angle rotation
        # t: (3,) translation
        # c: (2,) principal point
        # v: (N, 3) 3D points

        # 1) Axis-angle to rotation matrix
        angle = ch.sqrt(ch.sum(self.rt ** 2))
        eps = 1e-8
        axis = self.rt / (angle + eps)
        K = ch.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = ch.eye(3) + ch.sin(angle) * K + (1 - ch.cos(angle)) * (K.dot(K))

        # 2) Transform points
        v_cam = self.v.dot(R.T) + self.t  # (N,3)

        # 3) Perspective projection
        x = v_cam[:, 0] / v_cam[:, 2]
        y = v_cam[:, 1] / v_cam[:, 2]

        u = self.f[0] * x + self.c[0]
        v = self.f[1] * y + self.c[1]

        return ch.vstack((u, v)).T  # (N,2)
