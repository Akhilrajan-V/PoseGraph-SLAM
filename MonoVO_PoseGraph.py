import os
import cv2
import argparse
import numpy as np

from geometric_verifier import GeometricVerifier
from feature_extractor import FeatureExtractor
from visual_odometry import VisualOdometry
from loop_detector import LoopDetector
from pose_graph import PoseGraph
from keyframe import Keyframe
from utils import *

# Configuration
KEYFRAME_DISTANCE = 1.0
MIN_LOOP_SEPERATION = 10
MIN_INLIER = 70
MIN_INLIER_RATIO = 0.75

# Main
def main():    
    # Command-line arguments  
    parser = argparse.ArgumentParser(description="Monocular Visual SLAM - KITTI Sequence 09")
    parser.add_argument("--dataset-path", required=True, help="Path to the KITTI image directory.")
    parser.add_argument("--calibration-path", required=True, help="Path to the KITTI calibration file.")
    parser.add_argument("--ground-truth-path", required=True, help="Path to the ground-truth poses file.")
    parser.add_argument("--num-images", type=int, default=200, help=( "Number of images to process. " "Use 0 for all images." ))
    parser.add_argument("--iterations", type=int, default=10, help=( "Number of pose graph optimization iterations." ))
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    GROUND_TRUTH_PATH = args.ground_truth_path
    K = load_kitti_calibration(args.calibration_path, camera_id=0)

    # Load image filenames 
    image_files = sorted([ f for f in os.listdir(DATASET_PATH) if f.endswith(".png") ], key=lambda f: int(os.path.splitext(f)[0]))

    if len(image_files) < 2:
        raise RuntimeError("Not enough images found.")
    
    # Number of frames  
    if args.num_images <= 0:
        num_images = len(image_files)

    else:
        num_images = min(args.num_images, len(image_files))

    image_files = image_files[ :num_images ]
    print(f"Found {len(image_files)} images")
    print(f"Processing {len(image_files)} images")
    print(f"Pose graph iterations: " f"{args.iterations}")
  
    # Load ground truth
    ground_truth_poses = ( load_ground_truth_poses(GROUND_TRUTH_PATH, len(image_files)) )
    ground_truth_trajectory = np.array([ T[:3, 3].copy() for T in ground_truth_poses ])
   
    # Initialize modules    
    feature_extractor = FeatureExtractor()
    visual_odometry = VisualOdometry(K)
    pose_graph = PoseGraph()
    loop_detector = LoopDetector(min_seperation=MIN_LOOP_SEPERATION)
    geometric_verifier = GeometricVerifier(K=K, min_inlier=MIN_INLIER, min_inlier_ratio=MIN_INLIER_RATIO)
    
    # Initial pose 
    T_world_camera = np.eye(4, dtype=np.float64)

    keyframes = []
    # First frame
    first_frame_id = int(os.path.splitext(image_files[0])[0])
    previous_image_path = os.path.join(DATASET_PATH, image_files[0])
    previous_image = cv2.imread(previous_image_path, cv2.IMREAD_GRAYSCALE)

    if previous_image is None:
        raise RuntimeError(f"Could not read " f"{previous_image_path}")

    # First frame features
    temp_kp1, temp_desc1 = (feature_extractor.extract(previous_image))

    # First keyframe
    first_keyframe = Keyframe(first_frame_id, previous_image.copy(), temp_kp1, temp_desc1, T_world_camera.copy())
    keyframes.append(first_keyframe)

    # First pose graph node
    pose_graph.addNode(first_frame_id, T_world_camera.copy())
    print(f"Initial Keyframe: " f"{first_frame_id}")

    # Process image sequence
    for image_name in image_files[1:]:
        frame_id = int(os.path.splitext(image_name)[0])
        current_image_path = os.path.join(DATASET_PATH, image_name)
        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)

        if current_image is None:
            print(f"Warning: could not read " f"{current_image_path}")
            continue

        # 1. Extract and match features
        kp1, desc1 = (feature_extractor.extract(previous_image))
        kp2, desc2 = (feature_extractor.extract(current_image))
        pts1, pts2 = feature_extractor.match(kp1, desc1, kp2, desc2)

        # 2. Estimate relative motion
        vo_success = False

        if len(pts1) >= 8:
            try:
                R, t = (visual_odometry.estimateMotion(pts1, pts2))
                T_current_previous = (create_transform(R, t))

                # X_current =
                #     T_current_previous X_previous
                #
                # Therefore:
                #
                # T_world_current =
                #     T_world_previous
                #     @ inverse(T_current_previous)

                T_previous_current = (inverse_transform(T_current_previous))
                T_world_camera = (T_world_camera @ T_previous_current)
                vo_success = True

            except Exception as exc:
                print(f"VO failed at frame " f"{frame_id}: {exc}")

        else:
            print(f"Frame {frame_id:04d} | " f"not enough feature matches")

        # 3. Add EVERY frame to pose graph
        #
        # This is important.
        #
        # Every image now has a graph node.
        #
        # Therefore:
        #
        # N images
        #   ↓
        # N pose nodes
        #
        # This makes the optimized trajectory directly
        # comparable to ground truth for every processed frame.

        pose_graph.addNode(frame_id, T_world_camera.copy())

        # 4. Add sequential VO edge
        if vo_success:
            pose_graph.addEdge(frame_id - 1, frame_id, T_current_previous.copy())

        else:
            # If VO failed, maintain the current pose.
            #
            # Identity edge means:
            #
            # T_current_previous = I

            identity_motion = np.eye(4, dtype=np.float64)
            pose_graph.addEdge(frame_id - 1, frame_id, identity_motion)

        # 5. Keyframe selection
        current_position = ( T_world_camera[:3, 3] )
        last_keyframe_position = ( keyframes[-1].pose[:3, 3] )
        distance = np.linalg.norm(current_position - last_keyframe_position)

        # Create new keyframe
        if ( distance > KEYFRAME_DISTANCE ):
            new_keyframe = Keyframe(frame_id, current_image.copy(), kp2, desc2, T_world_camera.copy())
            keyframes.append(new_keyframe)

            print(f"\nNew Keyframe: " f"{frame_id} | " f"Distance = " f"{distance:.2f} m")

            # 6. Loop closure detection
            loop_candidates = (loop_detector.find_candidates(new_keyframe, keyframes[:-1]))
            loop_found = False

            for candidate, matches in loop_candidates:
                if ( new_keyframe.frame_id - candidate.frame_id <= MIN_LOOP_SEPERATION ):
                    continue

                print(f"Testing loop candidate " f"{candidate.frame_id} | " f"Matches = " f"{len(matches)}")
                result = (geometric_verifier.verify(new_keyframe, candidate, matches))

                if result is None:
                    continue

                # Verified loop closure
                print(f"LOOP CLOSURE DETECTED: " f"{new_keyframe.frame_id} " f"<-> " f"{candidate.frame_id}")
                print(f"Inliers: " f"{result['inlier_count']}")
                print(f"Inlier Ratio: " f"{result['inlier_ratio']:.2f}")

                # Loop R,t
                R_loop = result["R"]
                t_loop = ( result["t"] .reshape(3) )

                # Monocular translation scale
                old_position = ( candidate.pose[:3, 3] )
                new_position = ( new_keyframe.pose[:3, 3] )
                estimated_distance = ( np.linalg.norm(new_position - old_position) )
                t_norm = np.linalg.norm(t_loop)

                if ( t_norm < 1e-8 or estimated_distance < 1e-8 ):
                    print("Could not determine " "loop translation scale.")
                    continue

                t_loop = ( t_loop / t_norm ) * estimated_distance

                # Construct loop SE(3)
                T_loop = create_transform(R_loop, t_loop)

                # Loop ID
                loop_id = ( f"{candidate.frame_id}_" f"{new_keyframe.frame_id}" )

                # Add loop closure edge
                pose_graph.addEdge(candidate.frame_id, new_keyframe.frame_id, T_loop, loop_id=loop_id)

                print(f"Loop edge added: " f"{candidate.frame_id} " f"-> " f"{new_keyframe.frame_id}")
                loop_found = True

                break            

            if not loop_found:
                print("No verified loop closure.")

        # Current position
        position = ( T_world_camera[:3, 3] )

        print(f"Frame {frame_id:04d} | " f"Position: " f"x={position[0]:8.3f}, " f"y={position[1]:8.3f}, " f"z={position[2]:8.3f} | " f"Matches={len(pts1)}")

        # Next frame
        previous_image = current_image

    # Pose graph summary
    print("\n========================================")
    print("Pose Graph Summary")
    print("========================================")
    print(f"Processed frames: " f"{len(image_files)}")
    print(f"Pose graph nodes: " f"{len(pose_graph.nodes)}")
    print(f"Pose graph edges: " f"{len(pose_graph.edges)}")

    loop_edges = [ edge for edge in pose_graph.edges if edge.loop_id is not None ]
    print(f"Loop closure edges: " f"{len(loop_edges)}")

    # Trajectory BEFORE optimization
    trajectory_before = ( extract_trajectory(pose_graph) )

    # Pose graph optimization
    print("\n========================================")
    print("Starting Pose Graph Optimization")
    print("========================================")

    error_history = []

    if ( len(pose_graph.nodes) > 1 and len(pose_graph.edges) > 0 ):
        error_history = ( pose_graph.optimize(iterations=args.iterations, damping=1e-6) )

    else:
        print("Not enough graph data " "for optimization.")

    # Trajectory AFTER optimization
    trajectory_after = ( extract_trajectory(pose_graph) )

    # Final trajectory comparison
    num_compare = min(len(trajectory_before), len(trajectory_after), len(ground_truth_trajectory))
    trajectory_before = ( trajectory_before[:num_compare] )
    trajectory_after = ( trajectory_after[:num_compare] )
    ground_truth_trajectory = ( ground_truth_trajectory[:num_compare] )

    print(f"\nFinal trajectory comparison " f"contains {num_compare} frames.")

    # Plot trajectory
    plot_trajectory(trajectory_before, trajectory_after, ground_truth_trajectory)

    # Plot optimization error
    plot_optimization_error(error_history)

if __name__ == "__main__":
    main()