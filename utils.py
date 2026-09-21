import os
import numpy as np
import matplotlib.pyplot as plt


def load_ground_truth_poses(poses_file, num_frames):
    """
    Load KITTI ground-truth poses.

    Each line contains:

        r11 r12 r13 tx
        r21 r22 r23 ty
        r31 r32 r33 tz

    as 12 values.

    Returns:
        List of 4x4 T_world_camera matrices.
    """

    if not os.path.exists(poses_file):
        raise FileNotFoundError(f"Ground truth file not found:\n{poses_file}")

    data = np.loadtxt(poses_file, dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 12:
        raise ValueError(f"Expected 12 values per pose, " f"but got {data.shape[1]}")

    num_frames = min(num_frames, len(data))

    ground_truth = []

    for row in data[:num_frames]:

        T = np.eye(4, dtype=np.float64)

        T[:3, :4] = row.reshape(3, 4)

        ground_truth.append(T)

    return ground_truth

def extract_trajectory(pose_graph):
    """
    Extract camera positions from all pose graph nodes.
    """

    trajectory = []

    for node in pose_graph.nodes:
        trajectory.append( node.pose[:3, 3].copy() )

    return np.asarray( trajectory, dtype=np.float64 )

def align_trajectory(estimated, ground_truth):
    """
    Align monocular estimated trajectory to ground truth
    using a 3D similarity transform:

        X_gt ≈ s * R * X_est + t

    This is necessary because monocular VO does not have
    metric scale.

    Returns:
        aligned trajectory
        scale
        rotation
        translation
    """

    n = min( len(estimated), len(ground_truth) )

    estimated = estimated[:n]
    ground_truth = ground_truth[:n]

    estimated_mean = np.mean(estimated, axis=0)

    ground_truth_mean = np.mean(ground_truth, axis=0)

    X = (estimated - estimated_mean)

    Y = (ground_truth - ground_truth_mean)

   
    # Cross-covariance   
    H = (X.T @ Y) / n
    U, S, Vt = np.linalg.svd(H)
    R = (Vt.T @ U.T)
    
    # Reflection correction
    if np.linalg.det(R) < 0:

        Vt[-1, :] *= -1
        R = (Vt.T @ U.T)

    # Scale
    variance = np.sum(X ** 2) / n

    if variance < 1e-12:
        scale = 1.0

    else:
        scale = (np.sum(S) / variance)

    # Translation
    translation = (ground_truth_mean - scale * (R @ estimated_mean))
    
    # Apply transformation
    aligned = ( scale * ( estimated @ R.T ) + translation )

    return ( aligned, scale, R, translation )

def plot_trajectory( trajectory_before, trajectory_after, ground_truth ):

    num_frames = min( len(trajectory_before), len(trajectory_after), len(ground_truth) )

    trajectory_before = ( trajectory_before[:num_frames] )

    trajectory_after = ( trajectory_after[:num_frames] )

    ground_truth = ( ground_truth[:num_frames] )

    # --------------------------------------------------------
    # Align both monocular trajectories independently
    # to ground truth.
    # --------------------------------------------------------

    aligned_before, _, _, _ = ( align_trajectory( trajectory_before, ground_truth ) )

    aligned_after, _, _, _ = ( align_trajectory( trajectory_after, ground_truth ) )

    plt.figure( figsize=(12, 9) )

    # ========================================================
    # BEFORE OPTIMIZATION
    # ========================================================

    plt.plot( aligned_before[:, 0], aligned_before[:, 2], color="tab:blue", linewidth=2, label="Before Optimization" )

    # Start / End
    plt.scatter( aligned_before[0, 0], aligned_before[0, 2], color="tab:blue", marker="o", s=100, label="Before Start" )

    plt.scatter( aligned_before[-1, 0], aligned_before[-1, 2], color="tab:blue", marker="X", s=100, label="Before End" )

    # ========================================================
    # AFTER OPTIMIZATION
    # ========================================================

    plt.plot( aligned_after[:, 0], aligned_after[:, 2], color="tab:orange", linewidth=2, label="After Optimization" )

    # Start / End
    plt.scatter( aligned_after[0, 0], aligned_after[0, 2], color="tab:orange", marker="o", s=100, label="After Start" )

    plt.scatter( aligned_after[-1, 0], aligned_after[-1, 2], color="tab:orange", marker="X", s=100, label="After End" )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    plt.plot( ground_truth[:, 0], ground_truth[:, 2], color="tab:green", linewidth=2, label="Ground Truth" )

    # Start / End
    plt.scatter( ground_truth[0, 0], ground_truth[0, 2], color="tab:green", marker="o", s=100, label="Ground Truth Start" )

    plt.scatter( ground_truth[-1, 0], ground_truth[-1, 2], color="tab:green", marker="X", s=100, label="Ground Truth End" )

    # ========================================================
    # Plot settings
    # ========================================================

    plt.xlabel( "X (m)" )

    plt.ylabel( "Z (m)" )

    plt.title( f"KITTI Sequence 09 - " f"Trajectory Comparison ({num_frames} Frames)" )

    plt.axis( "equal" )

    plt.grid( True )

    plt.legend()

    plt.tight_layout()

    plt.show()

def plot_optimization_error( error_history ):

    if error_history is None:
        print( "No optimization error history available." )
        return

    if len(error_history) == 0:
        print( "Optimization error history is empty." )
        return

    iterations = np.arange( 1, len(error_history) + 1 )

    plt.figure( figsize=(10, 6) )

    plt.plot( iterations, error_history, color="tab:red", linewidth=2, marker="o", markersize=4 )

    plt.xlabel( "Iteration" )

    plt.ylabel( "Total Pose Graph Error" )

    plt.title( "Pose Graph Optimization Error" )

    plt.grid( True )

    plt.tight_layout()

    plt.show()
def create_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def inverse_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def load_kitti_calibration(calib_file, camera_id=0):
    """
    Load KITTI camera calibration.

    Parameters
    ----------
    calib_file : str Path to calib.txt

    camera_id : int
        Camera index.
        0 -> image_0
        1 -> image_1
        2 -> image_2
        3 -> image_3

    Returns
    -------
    K : np.ndarray
        3x3 camera intrinsic matrix
    """
    with open(calib_file, "r") as file:

        for line in file:
            line = line.strip()
            if not line:
                continue

            key, values = line.split(":", 1)
            if key == f"P{camera_id}":
                P = np.array([float(x) for x in values.split()], dtype=np.float64).reshape(3, 4)
                K = P[:, :3]
                return K

    raise ValueError(f"P{camera_id} not found in {calib_file}")

def skew(v):
    """ 
    Convert a 3D vector into a skew-symmetric matrix. 
    v = [x, y, z] 

    [v]^ = [0 -z y]  
           [z 0 -x] 
           [-y x 0] 
    """
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def so3_Log(R):
    """
    Logarithm map:
        SO(3) -> R^3
    Input:
        R : 3x3 rotation matrix
    Output:
        omega : 3-element rotation vector
    """

    R = np.asarray( R, dtype=np.float64 ).reshape(3, 3)

    cos_theta = ( np.trace(R) - 1.0 ) / 2.0

    # Numerical safety
    cos_theta = np.clip( cos_theta, -1.0, 1.0 )

    theta = np.arccos( cos_theta )

    # Very small rotation

    if theta < 1e-8:

        return np.zeros( 3, dtype=np.float64 )

    # Rotation vector

    omega = ( theta / (2.0 * np.sin(theta)) ) * np.array([ R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1] ])

    # GUARANTEE shape = (3,)

    return np.asarray( omega, dtype=np.float64 ).reshape(3)

def se3_Log(T):
    """
    Logarithm map:

        SE(3) -> R^6

    Output ordering:

        [rho_x, rho_y, rho_z,
         omega_x, omega_y, omega_z]
    """

    T = np.asarray( T, dtype=np.float64 ).reshape(4, 4)

    R = T[:3, :3]

    t = T[:3, 3].reshape(3)

    # Rotation logarithm

    omega = so3_Log(R)

    # GUARANTEE omega is (3,)
    omega = np.asarray( omega, dtype=np.float64 ).reshape(3)

    theta = np.linalg.norm( omega )

    # V^-1
    #
    # t = V rho
    #
    # rho = V^-1 t

    if theta < 1e-8:

        V_inv = np.eye( 3, dtype=np.float64 )

    else:

        Omega = skew( omega )

        half_theta = ( theta / 2.0 )

        cot_half_theta = ( 1.0 / np.tan(half_theta) )

        V_inv = ( np.eye(3) - 0.5 * Omega + ( 1.0 / theta**2 * ( 1.0 - half_theta * cot_half_theta ) ) * (Omega @ Omega) )

    # Translation logarithm

    rho = V_inv @ t

    # GUARANTEE rho is (3,)
    rho = np.asarray( rho, dtype=np.float64 ).reshape(3)

    # Final 6D vector

    xi = np.concatenate([ rho, omega ])

    return xi.reshape(6)

def so3_Exp(omega):
    """
    Exponential map:

        R^3 -> SO(3)

    omega:
        3D rotation vector

    Returns:
        3x3 rotation matrix
    """

    omega = np.asarray(omega, dtype=np.float64).reshape(3)

    theta = np.linalg.norm(omega)

    # Create skew matrix ONCE
    Omega = skew(omega)

    # Small-angle approximation
    if theta < 1e-8:

        return ( np.eye(3) + Omega + 0.5 * (Omega @ Omega) )

    A = np.sin(theta) / theta

    B = ( 1.0 - np.cos(theta) ) / (theta ** 2)

    R = ( np.eye(3) + A * Omega + B * (Omega @ Omega) )

    return R

def se3_Exp(xi):
    """
    Exponential map:

        R^6 -> SE(3)

    xi ordering:

        [rho_x, rho_y, rho_z,
         omega_x, omega_y, omega_z]

    Returns:
        4x4 SE(3) transformation
    """

    # Make absolutely sure xi is a 6-vector
    xi = np.asarray( xi, dtype=np.float64 ).reshape(6)

    # Split translation and rotation components

    rho = xi[:3]

    # IMPORTANT:
    # omega is still a VECTOR here.
    # Do NOT call skew() here.
    omega = xi[3:]

    theta = np.linalg.norm(omega)

    # Rotation
    #
    # so3_Exp() is responsible for creating Omega.

    R = so3_Exp(omega)

    # V matrix
    #
    # t = V @ rho

    Omega = skew(omega)

    if theta < 1e-8:

        V = ( np.eye(3) + 0.5 * Omega + (1.0 / 6.0) * (Omega @ Omega) )

    else:

        A = ( 1.0 - np.cos(theta) ) / (theta ** 2)

        B = ( theta - np.sin(theta) ) / (theta ** 3)

        V = ( np.eye(3) + A * Omega + B * (Omega @ Omega) )

    # Translation

    t = V @ rho

    # Construct SE(3)

    T = np.eye( 4, dtype=np.float64 )

    T[:3, :3] = R

    T[:3, 3] = t

    return T

def pose_graph_residual( pose_i, pose_j, measurement ):
    """
    Compute pose graph residual.

    T_i_j_pred =
        inverse(T_world_i) @ T_world_j

    Error:

        E =
            inverse(measurement)
            @ T_i_j_pred

    Residual:

        e = Log(E)
    """

    T_i_j_pred = ( inverse_transform(pose_i) @ pose_j )

    error_transform = ( inverse_transform(measurement) @ T_i_j_pred )

    return se3_Log( error_transform )

def numerical_edge_jacobians( pose_i, pose_j, measurement, epsilon=1e-6 ):
    """
    Compute numerical Jacobians for one pose-graph edge.

    Returns:

        J_i : 6x6
        J_j : 6x6

    Pose perturbation:

        T_new = Exp(delta) @ T

    delta:

        [dx, dy, dz, d_rx, d_ry, d_rz]
    """

    J_i = np.zeros( (6, 6), dtype=np.float64 )

    J_j = np.zeros( (6, 6), dtype=np.float64 )

    # Jacobian with respect to pose i

    for k in range(6):

        # IMPORTANT:
        # Use a 1-D vector, NOT (6,1)
        delta = np.zeros( 6, dtype=np.float64 )

        delta[k] = epsilon

        # Positive perturbation

        T_i_plus = ( se3_Exp(delta) @ pose_i )

        e_plus = pose_graph_residual( T_i_plus, pose_j, measurement )

        # Negative perturbation

        T_i_minus = ( se3_Exp(-delta) @ pose_i )

        e_minus = pose_graph_residual( T_i_minus, pose_j, measurement )

        # Central finite difference

        J_i[:, k] = ( e_plus - e_minus ) / (2.0 * epsilon)

    # Jacobian with respect to pose j

    for k in range(6):

        # IMPORTANT:
        # Use a 1-D vector, NOT (6,1)
        delta = np.zeros( 6, dtype=np.float64 )

        delta[k] = epsilon

        # Positive perturbation

        T_j_plus = ( se3_Exp(delta) @ pose_j )

        e_plus = pose_graph_residual( pose_i, T_j_plus, measurement )

        # Negative perturbation

        T_j_minus = ( se3_Exp(-delta) @ pose_j )

        e_minus = pose_graph_residual( pose_i, T_j_minus, measurement )

        # Central finite difference
        J_j[:, k] = ( e_plus - e_minus ) / (2.0 * epsilon)

    return J_i, J_j