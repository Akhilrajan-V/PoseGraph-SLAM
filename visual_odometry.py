import cv2
import numpy as np

class VisualOdometry:

    def __init__(self, K):
        self.K = K

    def estimateMotion(self, pts1, pts2):
        E, _ = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
        return R, t