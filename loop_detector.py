import cv2

class LoopDetector:

    def __init__(self, min_seperation=20):
        self.min_loop_seperation = min_seperation
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def find_candidates(self, curr_keyframe, keyframes):

        candidates = []
        for frame in keyframes:
            if (curr_keyframe.frame_id - frame.frame_id > self.min_loop_seperation and
                curr_keyframe.keypoints is not None and
                curr_keyframe.descriptors is not None):

                matches = self.feature_matcher.knnMatch(curr_keyframe.descriptors, frame.descriptors, k=2)
                good_matches = []

                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        good_matches.append(m)
                candidates.append((frame, good_matches))  

            candidates.sort(key=lambda x: len(x[1]), reverse=True)

        return candidates[:5]