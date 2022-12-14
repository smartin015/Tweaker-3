# -*- coding: utf-8 -*-
import numpy as np
from math import cos

class Orientation:
    def __init__(self, vec, weight=0):
        self.weight = weight
        self.vec = vec
        n = np.linalg.norm(vec) 
        if n != 1.0:
            self.vec /= n 

    def __eq__(self, other):
        return np.array_equal(self.vec, other.vec) and self.weight == other.weight

    def __repr__(self):
        return f"Orientation({self.vec}, {self.weight})"

    @staticmethod
    def supplements():
        """Supplement 18 additional vectors - axes plus 45 degree multi-axis rotations
        Returns:
            Basic Orientation Field"""
        
        # Cardinal directions
        result = []
        for axis in range(3):
            for unit in (1, -1):
               v = np.zeros(3)
               v[axis] = unit
               result.append(Orientation(v))

        tilt = 0.70710678 # unit vector when value of two axes, 3rd axis zero
        for u1 in (1, -1):
            for u2 in (1, -1):
                for xyz in [(0, 1), (0, 2), (1, 2)]:
                    v = np.zeros(3)
                    v[xyz[0]] = u1
                    v[xyz[1]] = u2
                    result.append(Orientation(v))

        return result

    @staticmethod
    def decimate(orients, degrees=5):
        """
        Removing similar orientations
        Args:
            orients (list): list of Orientations
            degrees (float): Angle within which similar orientations are removed
        Returns:
            Unique Orientations no closer than `degrees` between each other """
        tol_dist = cos(degrees * np.pi / 180)
        result = list()
        for i in orients:
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

    def euler(self, vector_tol):
        """Calculate euler rotation parameters and rotational matrix.
        Args:
            vector_tol (float): tolerance of vertical orientations (close to 0)
        Returns:
            rotation axis, rotation angle, 3x3 rotational matrix.
        """
        v = self.vec
        if v[0] ** 2 + v[1] ** 2 + (v[2] + 1.) ** 2 < abs(vector_tol):
            # If oriented z-down within tolerance
            r = [1., 0., 0.]
            phi = np.pi
        elif v[0] ** 2 + v[1] ** 2 + (v[2] - 1.) ** 2 < abs(vector_tol):
            # Basically z-up within tolerance
            r = [1., 0., 0.]
            phi = 0.
        else:
            # Get rotation axis and angle when axis is fixed to z=0
            phi = np.pi - np.arccos(-v[2])
            r = np.array(
                [-v[1], v[0], 0.])
            r /= np.linalg.norm(r)  # normalize

        # Construct the rotation matrix
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        m = np.empty((3, 3), dtype=np.float64)
        m[0, 0] = r[0] * r[0] * (1 - cos_phi) + cos_phi
        m[0, 1] = r[0] * r[1] * (1 - cos_phi) - r[2] * sin_phi
        m[0, 2] = r[0] * r[2] * (1 - cos_phi) + r[1] * sin_phi
        m[1, 0] = r[1] * r[0] * (1 - cos_phi) + r[2] * sin_phi
        m[1, 1] = r[1] * r[1] * (1 - cos_phi) + cos_phi
        m[1, 2] = r[1] * r[2] * (1 - cos_phi) - r[0] * sin_phi
        m[2, 0] = r[2] * r[0] * (1 - cos_phi) - r[1] * sin_phi
        m[2, 1] = r[2] * r[1] * (1 - cos_phi) + r[0] * sin_phi
        m[2, 2] = r[2] * r[2] * (1 - cos_phi) + cos_phi

        return r, phi, m

