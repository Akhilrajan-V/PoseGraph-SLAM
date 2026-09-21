import cv2
import numpy as np

class GeometricVerifier:
    def __init__(self, K, min_inlier=45, min_inlier_ratio=0.7):
        self.K = K
        self.min_inlier = min_inlier
        self.inlier_ratio = min_inlier_ratio

    def verify(self, cur_keyframe, old_keyframe, matches):

        if len(matches)<8:
            return None # not enough correspondence

        cur_pts = np.float32([cur_keyframe.keypoints[m.queryIdx].pt for m in matches])
        old_pts = np.float32([old_keyframe.keypoints[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(cur_pts, old_pts, self.K, cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None or mask is None:
            return None

        mask = mask.ravel()

        inlier_count = int(np.sum(mask)) 
        inlier_ratio = inlier_count/len(mask)

        if inlier_ratio < self.inlier_ratio or inlier_count < self.min_inlier:
            return None

        _, R, t, pose_mask = cv2.recoverPose(E, cur_pts, old_pts, self.K, mask.reshape(-1, 1))

        return {"inlier_count": inlier_count, "inlier_ratio":inlier_ratio, "E": E, "R": R, "t": t, "mask": mask}