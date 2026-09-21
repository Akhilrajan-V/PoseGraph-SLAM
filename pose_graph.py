import numpy as np
from utils import numerical_edge_jacobians, pose_graph_residual, se3_Exp

class PoseGraphNode:
    def __init__(self, node_id, pose):
        self.node_id = node_id
        self.pose = pose.copy()

class PoseGraphEdge:    
    def __init__(self, from_id, to_id, edge_transform, loop_id=None):
        self.from_id = from_id
        self.to_id = to_id
        self.transform = edge_transform.copy()
        self.loop_id = loop_id

class PoseGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def addNode(self, node_id, pose):
        node = PoseGraphNode(node_id, pose)
        self.nodes.append(node)

    def addEdge(self, from_id, to_id, edge_transform, loop_id=None):
        edge = PoseGraphEdge(from_id, to_id, edge_transform, loop_id)
        self.edges.append(edge)

    def get_node(self, node_id):
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def compute_edge_residual(self, edge): 
        node_i = self.get_node(edge.from_id) 
        node_j = self.get_node(edge.to_id) 

        if node_i is None or node_j is None: 
            raise ValueError( f"Invalid edge: " f"{edge.from_id} -> {edge.to_id}" ) 
        return pose_graph_residual( node_i.pose, node_j.pose, edge.transform )


    def optimize(self, iterations=10, damping=1e-6):

        if len(self.nodes) < 2:
            return []

        if len(self.edges) == 0:
            return []

        fixed_node_id = self.nodes[0].node_id

        variable_nodes = [
            node
            for node in self.nodes
            if node.node_id != fixed_node_id
        ]

        node_index = {}

        for index, node in enumerate(variable_nodes):
            node_index[node.node_id] = index

        num_variables = len(variable_nodes)

        state_size = 6 * num_variables

        error_history = []

        for iteration in range(iterations):

            H = np.zeros(
                (state_size, state_size),
                dtype=np.float64
            )

            b = np.zeros(
                state_size,
                dtype=np.float64
            )

            total_error = 0.0

            for edge in self.edges:

                node_i = self.get_node(edge.from_id)
                node_j = self.get_node(edge.to_id)

                if node_i is None or node_j is None:
                    continue

                e = pose_graph_residual(
                    node_i.pose,
                    node_j.pose,
                    edge.transform
                )

                J_i, J_j = numerical_edge_jacobians(
                    node_i.pose,
                    node_j.pose,
                    edge.transform
                )

                total_error += (
                    0.5 * np.dot(e, e)
                )

                i_is_variable = (
                    edge.from_id in node_index
                )

                j_is_variable = (
                    edge.to_id in node_index
                )

                if i_is_variable:

                    i_index = node_index[
                        edge.from_id
                    ]

                    i_start = 6 * i_index
                    i_end = i_start + 6

                    H[
                        i_start:i_end,
                        i_start:i_end
                    ] += J_i.T @ J_i

                    b[
                        i_start:i_end
                    ] += J_i.T @ e

                if j_is_variable:

                    j_index = node_index[
                        edge.to_id
                    ]

                    j_start = 6 * j_index
                    j_end = j_start + 6

                    H[
                        j_start:j_end,
                        j_start:j_end
                    ] += J_j.T @ J_j

                    b[
                        j_start:j_end
                    ] += J_j.T @ e

                if (
                    i_is_variable
                    and j_is_variable
                ):

                    i_index = node_index[
                        edge.from_id
                    ]

                    j_index = node_index[
                        edge.to_id
                    ]

                    i_start = 6 * i_index
                    i_end = i_start + 6

                    j_start = 6 * j_index
                    j_end = j_start + 6

                    H[
                        i_start:i_end,
                        j_start:j_end
                    ] += J_i.T @ J_j

                    H[
                        j_start:j_end,
                        i_start:i_end
                    ] += J_j.T @ J_i

            # ----------------------------------------------------
            # Store error for plotting
            # ----------------------------------------------------

            error_history.append(
                total_error
            )

            # ----------------------------------------------------
            # Damping
            # ----------------------------------------------------

            H += (
                damping
                * np.eye(
                    state_size,
                    dtype=np.float64
                )
            )

            # ----------------------------------------------------
            # Solve
            # ----------------------------------------------------

            try:

                delta = np.linalg.solve(
                    H,
                    -b
                )

            except np.linalg.LinAlgError:

                delta = np.linalg.lstsq(
                    H,
                    -b,
                    rcond=None
                )[0]

            # ----------------------------------------------------
            # Update poses
            # ----------------------------------------------------

            max_update = 0.0

            for node in variable_nodes:

                index = node_index[
                    node.node_id
                ]

                start = 6 * index
                end = start + 6

                delta_i = delta[
                    start:end
                ]

                max_update = max(
                    max_update,
                    np.linalg.norm(delta_i)
                )

                node.pose = (
                    se3_Exp(delta_i)
                    @ node.pose
                )

            print(
                f"Iteration {iteration + 1}: "
                f"error = {total_error:.6f}, "
                f"max update = {max_update:.6e}"
            )

            if max_update < 1e-6:

                print(
                    "Pose graph optimization converged."
                )

                break

        return error_history