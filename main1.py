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
    "C:/Users/vakhi/Downloads/Vision_Dataset/dataset/sequences/09/calib.txt",
    camera_id=0
)


def main():

    # ========================================================
    # Load image filenames
    # ========================================================

    image_files = sorted(
        [
            f
            for f in os.listdir(DATASET_PATH)
            if f.endswith(".png")
        ],
        key=lambda f: int(
            os.path.splitext(f)[0]
        )
    )

    if len(image_files) < 2:
        raise RuntimeError("Not enough images found.")

    print(f"Found {len(image_files)} images")

    # ========================================================
    # Initialize modules
    # ========================================================

    KEYFRAME_DISTANCE = 1.25

    MIN_LOOP_SEPERATION = 20

    feature_extractor = FeatureExtractor()

    visual_odometry = VisualOdometry(K)

    pose_graph = PoseGraph()

    loop_detector = LoopDetector(
        min_seperation=MIN_LOOP_SEPERATION
    )

    geometric_verifier = GeometricVerifier(
        K=K,
        min_inlier=55,
        min_inlier_ratio=0.55
    )

    # ========================================================
    # Initial camera pose and keyframes
    # ========================================================

    T_world_camera = np.eye(4)

    keyframes = []


    # ========================================================
    # First frame
    # ========================================================

    first_frame_id = int(
        os.path.splitext(image_files[0])[0]
    )

    pose_graph.addNode(
        first_frame_id,
        T_world_camera.copy()
    )

    previous_image_path = os.path.join(
        DATASET_PATH,
        image_files[0]
    )

    previous_image = cv2.imread(
        previous_image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if previous_image is None:
        raise RuntimeError(
            f"Could not read {previous_image_path}"
        )

    # --------------------------------------------------------
    # Initialize first keyframe
    # --------------------------------------------------------

    temp_kp1, temp_desc1 = (
        feature_extractor.extract(
            previous_image
        )
    )

    first_keyframe = Keyframe(
        first_frame_id,
        previous_image.copy(),
        temp_kp1,
        temp_desc1,
        T_world_camera.copy()
    )

    keyframes.append(
        first_keyframe
    )

    print(
        f"Initial Keyframe: "
        f"{first_frame_id}"
    )

    # ========================================================
    # Process image sequence
    # ========================================================

    for image_name in image_files[1:200]:

        frame_id = int(
            os.path.splitext(image_name)[0]
        )

        current_image_path = os.path.join(
            DATASET_PATH,
            image_name
        )

        current_image = cv2.imread(
            current_image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if current_image is None:

            print(
                f"Warning: could not read "
                f"{current_image_path}"
            )

            continue

        # ====================================================
        # 1. Extract and match features
        # ====================================================

        kp1, desc1 = (
            feature_extractor.extract(
                previous_image
            )
        )

        kp2, desc2 = (
            feature_extractor.extract(
                current_image
            )
        )

        pts1, pts2 = feature_extractor.match(
            kp1,
            desc1,
            kp2,
            desc2
        )

        # ----------------------------------------------------
        # Check that enough matches exist
        # ----------------------------------------------------

        if len(pts1) < 8:

            print(
                f"Frame {frame_id:04d} | "
                f"File {image_name} | "
                f"not enough feature matches"
            )

            previous_image = current_image

            continue

        # ====================================================
        # 2. Estimate relative camera motion
        # ====================================================
        #
        # VisualOdometry returns:
        #
        # X_current =
        #       R @ X_previous + t
        #
        # Therefore:
        #
        # T_current_previous
        #
        # maps previous-camera coordinates
        # into current-camera coordinates.
        #
        # ====================================================

        R, t = visual_odometry.estimateMotion(
            pts1,
            pts2
        )

        # ====================================================
        # 3. Construct relative transformation
        # ====================================================

        T_current_previous = create_transform(
            R,
            t
        )

        # ====================================================
        # 4. Convert VO motion into global pose
        # ====================================================
        #
        # We store:
        #
        # X_world =
        #       T_world_camera @ X_camera
        #
        # Since:
        #
        # X_current =
        #       T_current_previous @ X_previous
        #
        # we need:
        #
        # X_previous =
        #       inverse(T_current_previous)
        #       @ X_current
        #
        # Therefore:
        #
        # T_world_current =
        #       T_world_previous
        #       @ inverse(T_current_previous)
        #
        # ====================================================

        T_previous_current = (
            inverse_transform(
                T_current_previous
            )
        )

        T_world_camera = (
            T_world_camera
            @ T_previous_current
        )

        # ====================================================
        # 5. Keyframe selection
        # ====================================================

        current_position = (
            T_world_camera[:3, 3]
        )

        last_keyframe_position = (
            keyframes[-1].pose[:3, 3]
        )

        distance = np.linalg.norm(
            current_position
            - last_keyframe_position
        )

        # ====================================================
        # Create new keyframe
        # ====================================================

        if distance > KEYFRAME_DISTANCE:

            new_keyframe = Keyframe(
                frame_id,
                current_image.copy(),
                kp2,
                desc2,
                T_world_camera.copy()
            )

            keyframes.append(
                new_keyframe
            )

            print(
                f"\nNew Keyframe: "
                f"{frame_id} | "
                f"Distance = {distance:.2f} m"
            )

            # =================================================
            # 5A. Add keyframe node to pose graph
            # =================================================

            pose_graph.addNode(
                frame_id,
                T_world_camera.copy()
            )

            # =================================================
            # 5B. Add sequential/keyframe odometry edge
            # =================================================
            #
            # IMPORTANT:
            #
            # Our residual expects:
            #
            # measurement =
            #       T_i^-1 @ T_j
            #
            # Therefore the measurement must be:
            #
            # T_previous_keyframe_current_keyframe
            #
            # NOT T_current_previous.
            #
            # =================================================

            previous_keyframe = keyframes[-2]

            T_previous_keyframe_current_keyframe = (
                inverse_transform(
                    previous_keyframe.pose
                )
                @ new_keyframe.pose
            )

            pose_graph.addEdge(
                previous_keyframe.frame_id,
                new_keyframe.frame_id,
                T_previous_keyframe_current_keyframe
            )

            # =================================================
            # 5C. Check for loop closure
            # =================================================

            loop_candidates = (
                loop_detector.find_candidates(
                    new_keyframe,
                    keyframes[:-1]
                )
            )

            loop_found = False

            for candidate, matches in loop_candidates:

                if (
                    new_keyframe.frame_id
                    - candidate.frame_id
                    <= MIN_LOOP_SEPERATION
                ):
                    continue

                print(
                    f"Testing loop candidate "
                    f"{candidate.frame_id} | "
                    f"Matches = {len(matches)}"
                )

                result = (
                    geometric_verifier.verify(
                        new_keyframe,
                        candidate,
                        matches
                    )
                )

                if result is None:
                    continue

                # =============================================
                # Verified loop closure
                # =============================================

                print(
                    f"LOOP CLOSURE DETECTED: "
                    f"{new_keyframe.frame_id} "
                    f"<-> "
                    f"{candidate.frame_id}"
                )

                print(
                    f"Inliers: "
                    f"{result['inlier_count']}"
                )

                print(
                    f"Inlier Ratio: "
                    f"{result['inlier_ratio']:.2f}"
                )

                # =============================================
                # Get relative R and t
                # =============================================

                R_loop = result["R"]

                t_loop = (
                    result["t"]
                    .reshape(3)
                )

                # =============================================
                # Monocular translation scale
                # =============================================

                old_position = (
                    candidate.pose[:3, 3]
                )

                new_position = (
                    new_keyframe.pose[:3, 3]
                )

                estimated_distance = np.linalg.norm(
                    new_position
                    - old_position
                )

                t_norm = np.linalg.norm(
                    t_loop
                )

                if (
                    t_norm < 1e-8
                    or estimated_distance < 1e-8
                ):

                    print(
                        "Could not determine "
                        "loop translation scale."
                    )

                    continue

                t_loop = (
                    t_loop / t_norm
                ) * estimated_distance

                # =============================================
                # Construct loop SE(3)
                # =============================================

                T_loop = create_transform(
                    R_loop,
                    t_loop
                )

                # =============================================
                # Add loop closure edge
                # =============================================
                #
                # Geometric verification gives:
                #
                # X_old =
                #       R_loop @ X_current + t_loop
                #
                # Therefore:
                #
                # T_loop = T_old_current
                #
                # which matches:
                #
                # from_id = old keyframe
                # to_id   = current keyframe
                #
                # =============================================

                loop_id = (
                    f"{candidate.frame_id}_"
                    f"{new_keyframe.frame_id}"
                )

                pose_graph.addEdge(
                    candidate.frame_id,
                    new_keyframe.frame_id,
                    T_loop,
                    loop_id=loop_id
                )

                print(
                    f"Loop edge added: "
                    f"{candidate.frame_id} -> "
                    f"{new_keyframe.frame_id}"
                )

                loop_found = True

                break

            if not loop_found:

                print(
                    "No verified loop closure."
                )

        # ====================================================
        # Print current camera position
        # ====================================================

        position = (
            T_world_camera[:3, 3]
        )

        print(
            f"Frame {frame_id:04d} | "
            f"File {image_name} | "
            f"Position: "
            f"x={position[0]:8.3f}, "
            f"y={position[1]:8.3f}, "
            f"z={position[2]:8.3f} | "
            f"Matches={len(pts1)}"
        )

        # ====================================================
        # Prepare for next frame
        # ====================================================

        previous_image = current_image

    # ========================================================
    # Pose graph summary
    # ========================================================

    print("\n========================================")
    print("Pose Graph Summary")
    print("========================================")

    print(
        f"Nodes: {len(pose_graph.nodes)}"
    )

    print(
        f"Edges: {len(pose_graph.edges)}"
    )

    loop_edges = [
        edge
        for edge in pose_graph.edges
        if edge.loop_id is not None
    ]

    print(
        f"Loop Edges: {len(loop_edges)}"
    )

    # ========================================================
    # Extract trajectory BEFORE optimization
    # ========================================================

    trajectory_before = np.array(
        [
            node.pose[:3, 3].copy()
            for node in pose_graph.nodes
        ]
    )

    # ========================================================
    # Print loop edges
    # ========================================================

    print("\nLoop Closures:")

    for edge in loop_edges:

        print(
            f"  {edge.from_id} -> "
            f"{edge.to_id} "
            f"(loop_id={edge.loop_id})"
        )

    # ========================================================
    # Pose Graph Optimization
    # ========================================================

    print("\n========================================")
    print("Starting Pose Graph Optimization")
    print("========================================")

    if (
        len(pose_graph.nodes) > 1
        and len(pose_graph.edges) > 0
    ):

        pose_graph.optimize(
            iterations=250,
            damping=1e-6
        )

    else:

        print(
            "Not enough graph data "
            "for optimization."
        )

    # ========================================================
    # Extract trajectory AFTER optimization
    # ========================================================

    trajectory_after = np.array(
        [
            node.pose[:3, 3].copy()
            for node in pose_graph.nodes
        ]
    )

    # ========================================================
    # Plot before and after optimization
    # ========================================================

    plt.figure(
        figsize=(10, 8)
    )

    # --------------------------------------------------------
    # Before optimization
    # --------------------------------------------------------

    plt.plot(
        trajectory_before[:, 0],
        trajectory_before[:, 2],
        linewidth=2,
        label="Before Optimization"
    )

    # --------------------------------------------------------
    # After optimization
    # --------------------------------------------------------

    plt.plot(
        trajectory_after[:, 0],
        trajectory_after[:, 2],
        linewidth=2,
        label="After Optimization"
    )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    plt.scatter(
        trajectory_before[0, 0],
        trajectory_before[0, 2],
        s=80,
        label="Start"
    )

    # --------------------------------------------------------
    # End
    # --------------------------------------------------------

    plt.scatter(
        trajectory_after[-1, 0],
        trajectory_after[-1, 2],
        s=80,
        label="End"
    )

    plt.xlabel("X")

    plt.ylabel("Z")

    plt.title(
        "KITTI Sequence 09 - "
        "Pose Graph Optimization"
    )

    plt.axis("equal")

    plt.grid(True)

    plt.legend()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()