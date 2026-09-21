import numpy as np

class Keyframe:   
    def __init__(self, frame_id, image, keypoints, descriptors, pose):
        self.frame_id = frame_id
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.pose = pose