import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, n_features=2000):             
        # Use ORB (Oriented FAST and Rotated BRIEF) to detect keypoints and compute descriptors
        self.orb = cv2.ORB_create(nfeatures=n_features)

        # Create a BFMatcher object with Hamming distance
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract(self, image):    
        keypoints, descriptors = self.orb.detectAndCompute(image, None)        
        return keypoints, descriptors

    def match(self, kp1, desc1, kp2, desc2):        
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        good_matches = []      

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)         

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return pts1, pts2