import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from feature_extractor import FeatureExtractor
from visual_odometry import VisualOdometry
from pose_graph import PoseGraph
from keyframe import Keyframe
from loop_detector import LoopDetector
from geometric_verifier import GeometricVerifier
from utils import *

# ============================================================
# KITTI configuration
# ============================================================
DATASET_PATH = "C:/Users/vakhi/Downloads/Vision_Dataset/dataset/sequences/09/image_0"

# KITTI calibration for sequence 09
# We are keeping this here because main.py is currently
# responsible for the KITTI-specific setup.

K = load_kitti_calibration(
    "C:/Users/vakhi/Downloads/Vision_Dataset/dataset/sequences/09/calib.txt", camera_id=0)

def main():
    # Load image filenames
    image_files = sorted([ f for f in os.listdir(DATASET_PATH) if f.endswith(".png")])

    if len(image_files) < 2:
        raise RuntimeError("Not enough images")

    print(f"Found {len(image_files)} images")

    # Initialize modules
    feature_extractor = FeatureExtractor()
    visual_odometry = VisualOdometry(K)
    pose_graph = PoseGraph()
    loop_detector = LoopDetector(min_seperation=20)
    geometric_verifier = GeometricVerifier(K=K, min_inlier=65, min_inlier_ratio=0.75)

    # --------------------------------------------------------
    # Initial camera pose and keyframe 
    T_world_camera = np.eye(4)
    
    # Keyframe initialization
    keyframes = []
    KEYFRAME_DISTANCE = 1.25  # meters
    MIN_LOOP_SEPERSATION = 20  # frames

    # Add first node
    pose_graph.addNode(0, T_world_camera)

    # Load first image
    previous_image_path = os.path.join(DATASET_PATH, image_files[0])

    previous_image = cv2.imread(previous_image_path, cv2.IMREAD_GRAYSCALE)

    if previous_image is None:
        raise RuntimeError(f"Could not read {previous_image_path}")

    # initialize keyframe for the first frame
    temp_kp1, temp_desc1 = feature_extractor.extract(previous_image)
    first_keyframe = Keyframe(0, previous_image, temp_kp1, temp_desc1, T_world_camera.copy())
    keyframes.append(first_keyframe)

    # ========================================================
    # Process image sequence
    # ========================================================
    for frame_id, image_name in enumerate(image_files[1:], start=1):
        current_image_path = os.path.join(DATASET_PATH, image_name)

        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)

        if current_image is None:
            print(f"Warning: could not read {current_image_path}")
            continue

        # 1. Extract and match features
        kp1, desc1 = feature_extractor.extract(previous_image)
        kp2, desc2 = feature_extractor.extract(current_image)
        pts1, pts2 = feature_extractor.match(kp1, desc1, kp2, desc2)     

        # Check that enough matches exist
        if len(pts1) < 8:
            print(f"Frame {frame_id:04d} | File {image_name} | not enough feature matches")
            previous_image = current_image
            continue

        # ----------------------------------------------------
        # 2. Estimate relative camera motion
        # Returns:
        # X_current = R X_previous + t
        #
        # Therefore:
        # T_current_previous
        # ----------------------------------------------------
        R, t = visual_odometry.estimateMotion(pts1, pts2)
 
        # 3. Construct relative transformation
        T_current_previous = create_transform(R, t)

        # ----------------------------------------------------
        # 4. Convert relative VO motion into global pose
        #
        # VO gives:
        #
        # X_current = T_current_previous X_previous
        #
        # But our node pose is:
        # X_world = T_world_camera X_camera
        #
        # Therefore:
        # T_world_current = T_world_previous * inverse(T_current_previous)
        # ----------------------------------------------------

        T_previous_current = inverse_transform(T_current_previous)
        T_world_camera = (T_world_camera @ T_previous_current)

        # Check if a new keyframe should be created
        current_position = T_world_camera[:3, 3]
        last_keyframe_position = keyframes[-1].pose[:3, 3]

        distance = np.linalg.norm(current_position - last_keyframe_position)

        # if keyframe detect distance is satified add keyframe
        if distance > KEYFRAME_DISTANCE:
            new_keyframe = Keyframe(frame_id, current_image, kp2, desc2, T_world_camera.copy())
            keyframes.append(new_keyframe)
            print(f"New Keyframe: {frame_id} | Distance = {distance:.2f} m")

            # Check for loop closure
            loop_candidates = loop_detector.find_candidates(new_keyframe, keyframes[:-1])

            for candidate, matches in loop_candidates:
                result = geometric_verifier.verify(new_keyframe, candidate, matches)
                if result is not None:
                    print(f"Loop closure detected between frame {new_keyframe.frame_id} and frame {candidate.frame_id}")
                    print(f"Inliers: {result['inlier_count']}, Inlier Ratio: {result['inlier_ratio']:.2f}")

                    R_loop = result['R']
                    t_loop = result['t'].reshape(3)

                    # Monocular translation has unknown scale.
                    # Match it to the current visual trajectory scale
                    old_position = candidate.pose[:3, 3]
                    new_position = new_keyframe.pose[:3, 3]

                    estimated_distance = np.linalg.norm(current_position - old_position)
                    t_norm = np.linalg.norm(t_loop)

                    if t_norm > 1e-8:
                        t_loop = (t_loop/t_norm)*estimated_distance

                    # Construct SE(3)
                    T_Loop = np.eye(4, dtype=np.float32)

                    T_Loop[:3, :3] = R_loop
                    T_Loop[:3, 3] = t_loop

                    pose_graph.addEdge(candidate.frame_id, new_keyframe.frame_id, T_Loop, loop_id=new_keyframe.frame_id) 

                    break # Only consider the first valid loop closure

        # 5. Add new node to pose graph
        pose_graph.addNode(frame_id, T_world_camera.copy())

        # 6. Add sequential VO edge
        pose_graph.addEdge(frame_id-1, frame_id, T_current_previous.copy())

        # Print current position
        position = T_world_camera[:3, 3]

        print(
            f"Frame {frame_id:04d} | "
            f"File {image_name} | "
            f"Position: "
            f"x={position[0]:8.3f}, "
            f"y={position[1]:8.3f}, "
            f"z={position[2]:8.3f} | "
            f"Matches={len(pts1)}"
            )

        # Prepare for next frame
        previous_image = current_image

    # ========================================================
    # Extract trajectory from pose graph
    # ========================================================
    trajectory = []
    for node in pose_graph.nodes:
        position = node.pose[:3, 3]
        trajectory.append(position)
    trajectory = np.array(trajectory)

    # ========================================================
    # Plot trajectory
    # ========================================================
    plt.figure(figsize=(10, 8))

    plt.plot(trajectory[:, 0], trajectory[:, 2], linewidth=2)

    plt.scatter(trajectory[0, 0], trajectory[0, 2], s=80, label="Start")
    plt.scatter(trajectory[-1, 0], trajectory[-1, 2], s=80, label="End")

    plt.xlabel("X")
    plt.ylabel("Z")

    plt.title("KITTI Sequence 09 - Visual Odometry Trajectory")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.show()

# ====== ====== MAIN ====== ====== #
if __name__ == "__main__":
    main()