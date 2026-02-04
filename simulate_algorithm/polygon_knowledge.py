

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


class PolygonKnowledge:    
    """
    Data structure representing polygon geometry and derived properties.

    Assumptions:
        1. Corner coordinates are either fully known (x, y) or None.
        2. n_sides >= 3.
        3. Edges are ordered counter-clockwise.
        4. Edge i has tail at corner i and head at corner (i + 1) % n_sides.
        5. Interior corner angles are in degrees, range [0, 360].
        6. Surface lies on the left side of each directed edge.
        7. Minimum edge length is 10 cm.
    """

    def __init__(self, n_sides):
        
        self.n_sides: int = n_sides
        self.slopes: list[float | None] = [None] * n_sides
        self.lengths: list[float | None] = [None] * n_sides
        self.edge_unit_vectors: list[tuple[float, float] | None] = [None] * n_sides
        self.corners: list[tuple[float, float] | None] = [None] * n_sides
        self.is_reflexive_angle: list[bool | None] = [None] * n_sides
        self.corner_angles: list[float | None] = [None] * n_sides
        self.dihedrals: list[float | None] = [None] * n_sides
        self.internal_points_on_edge: list[list[tuple[float, float]]] = [
            [] for _ in range(n_sides)
        ]

    def print_knowledge(self, string_prefix: str = "") -> None:
        """
        Method to print the current knowledge of the polygon in a formatted way
        
        :param string_prefix: Prefix added before printing knowledge
        """
        print("\n*** " + string_prefix + " ***")
        for i in range(self.n_sides):
            # Handle each value with a conditional to check for None and numeric types
            slope = self.slopes[i] if self.slopes[i] is not None else "None"
            edge_unit_vectors = f"({self.edge_unit_vectors[i][0]:.4f}, {self.edge_unit_vectors[i][1]:.4f})" if self.edge_unit_vectors[i] is not None else "None"
            length = self.lengths[i] if self.lengths[i] is not None else "None"
            corner = f"({self.corners[i][0]:.4f}, {self.corners[i][1]:.4f})" if self.corners[i] is not None else "None"
            angle = self.corner_angles[i] if self.corner_angles[i] is not None else "None"
            dihedral = self.dihedrals[i] if self.dihedrals[i] is not None else "None"
            corner_angle_is_reflexive = self.is_reflexive_angle[i] if self.is_reflexive_angle[i] is not None else "None"

            # Check if numeric values should be formatted
            if isinstance(slope, (int, float)):
                slope = f"{slope:.4f}"
            if isinstance(length, (int, float)):
                length = f"{length:.4f}"
            if isinstance(angle, (int, float)):
                angle = f"{angle:.4f}"
            if isinstance(dihedral, (int, float)):
                dihedral = f"{dihedral:.4f}"

            # Handle points_on_edge, format each pair of coordinates if not None
            points_on_edge = [(f'{x:.4f}', f'{y:.4f}') for x, y in self.internal_points_on_edge[i]] if self.internal_points_on_edge[i] is not None else "None"

            # Print the formatted output
            print(f" Edge {i}: slope={slope}, edge_unit_vector={edge_unit_vectors}, length={length}, "
                f"corner={corner}, angle={angle}, dihedral={dihedral}, points_on_edge={points_on_edge}, is_corner_angle_reflexive={corner_angle_is_reflexive}")
        
    def plot_polygon(self, title: str = "Polygon Knowledge") -> None:
        """
        Method to plot the polygon based on current corner knowledge
        
        :param title: Title displayed on the plot
        """
        corners = self.corners
        
        # check if all corners are known
        all_corners_known = all(c is not None for c in corners)        
        if not all_corners_known:
            x, y = [], []
            print("Warning: Not all corners are known, plotting only known corners.")
            plt.figure(figsize=(6,6))
            # draw lines between known corners in order
            for i in range(len(corners)):
                x.append(corners[i][0] if corners[i] is not None else None)
                y.append(corners[i][1] if corners[i] is not None else None)
                if corners[i] is not None and corners[(i+1)%len(corners)] is not None:
                    plt.plot([corners[i][0], corners[(i+1)%len(corners)][0]], 
                             [corners[i][1], corners[(i+1)%len(corners)][1]], 
                             linestyle='--', color='gray')
                if len(self.internal_points_on_edge[i]) > 0:
                    for pt in self.internal_points_on_edge[i]:
                        plt.scatter(pt[0], pt[1], marker='x', color='r', s=100)
            plt.scatter(x, y, marker='o', color='r')
            plt.title(title + " (Partial)")
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True)
            plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal scaling for X and Y axes
            plt.show()
        else:
            x, y = zip(*corners)
            plt.figure(figsize=(6,6))
            plt.plot(x + (x[0],), y + (y[0],), marker='o', linestyle='-', color='b')  # Close the polygon
            plt.fill(x + (x[0],), y + (y[0],), 'b', alpha=0.2)  # Filling the polygon with some transparency
            for i in range(len(corners)):
                if len(self.internal_points_on_edge[i]) > 0:
                    for pt in self.internal_points_on_edge[i]:
                        plt.scatter(pt[0], pt[1], marker='x', color='r', s=100)
            plt.title(title)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True)
            plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal scaling for X and Y axes
            plt.show()
            
    def get_all_points_on_edge(self, i: int) -> Optional[List[Tuple[float, float]]]:
        """
        Method to get all points on an edge, including corners
        
        :param i: Index of edge
        :return: If any points exists, then return ordered points on the edge in counterclockwise
        """
        points = []
        internal_points = self.internal_points_on_edge[i]
        corner = self.corners[i]
        next_i = (i+1) % self.n_sides
        
        if corner is not None:
            corner_arr = np.asarray(corner)
            points.append(corner)
            
            if len(internal_points) > 1:
                key_fn = lambda p: np.linalg.norm(np.asarray(p) - corner_arr)
                internal_points.sort(key=key_fn)
        points.extend(internal_points)
        if self.corners[next_i]:
            points.append(self.corners[next_i])
        return points

