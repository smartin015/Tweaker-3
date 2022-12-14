# -*- coding: utf-8 -*-
import numpy as np

class Orientation:
    def __init__(self, vec, weight):
        self.weight = weight
        self.vec = vec / np.linalg.norm(vec)

    def __eq__(self, other):
        return np.array_equal(self.vec, other.vec) and self.weight == other.weight

    def __repr__(self):
        return f"Orientation({self.vec}, {self.weight})"

    @staticmethod
    def add_supplements():
        """Supplement 18 additional vectors - axes plus 45 degree multi-axis rotations
        Returns:
            Basic Orientation Field"""
        
        # Cardinal directions
        result = []
        for axis in range(3):
            for unit in (1, -1):
               v = np.zeros(3)
               v[axis] = unit
               result.append(Orientation(v, 0))

        tilt = 0.70710678 # unit vector when value of two axes, 3rd axis zero
        for u1 in (1, -1):
            for u2 in (1, -1):
                for xyz in [(0, 1), (0, 2), (1, 2)]:
                    v = np.zeros(3)
                    v[xyz[0]] = u1
                    v[xyz[1]] = u2
                    result.append(Orientation(v, 0))

        return result

    @staticmethod
    def unique(old_orients, degrees=5):
        """
        Removing similar orientations
        Args:
            old_orients (list): list of Orientations
        Returns:
            Unique Orientations no closer than `degrees` between each other """
        tol_dist = acos(np.degrees * np.pi / 180)
        result = list()
        for i in old_orients:
            duplicate = False
            for j in result:
                # redundant vectors have an difference smaller than
                # dist = ory-xor < tol_dist-> alpha = 5 degrees
                if np.dot(i.vec, j.vec) > tol_dist:
                    # since orientations are unit vectors, i dot j = cos(theta)
                    # when theta = 0, cos(theta) = 1, so we want to keep
                    # all values under tol_dist
                    duplicate = True
                    break
            if not duplicate:
                result.append(i)
        return result 

