# -*- coding: utf-8 -*-
import os
import re
import math
from time import time, sleep
from collections import Counter
# upgrade numpy with: "pip install numpy --upgrade"
from .orientation import Orientation
import numpy as np

class InvalidMeshException(Exception):
    pass

class M:
    NORM = 0
    V0 = 1
    V1 = 2
    V2 = 3
    A1 = 4
    A2 = 5

class A1:
    V0Z = 0
    V1Z = 1
    V2Z = 2

class A2:
    AREA = 0
    MAX_Z = 1
    MED_Z = 2

class V:
    X = 0
    Y = 1
    Z = 2

class Mesh:
    LIN_CONG_MULT = 127
    LIN_CONG_INCR = 8191

    @classmethod
    def _compute_face_normals(self, v0, v1, v2):
        # Returns face_count x 3 matrix of normal vectors, non-normalized
        row_number = v0.shape[0]
        return np.cross(np.subtract(v1, v0), np.subtract(v2, v0)) \
                .reshape(row_number, 1, 3)

    def __init__(self, content, negl_size=None):
        """The Mesh format gets preprocessed for a better performance and stored into self.mesh
        Args:
            content (np.array): undefined representation of the mesh
        Returns:
            mesh (np.array): with format face_count x 6 x 3, i.e.
                             face_count x (norm, v0, v1, v2, addendum1, addendum2) x (x, y, z) 
                             addendum1(x,y,z) = (v0_z, v1_z, v2_z)
                             addendum2(x,y,z) = (norm_magnitude, max_z, median_z)
        """
        mesh = np.array(content, dtype=np.float64)

        if len(mesh.shape) not in (2,3) or mesh.shape[0] == 0 or mesh.shape[1] == 0:
            raise InvalidMeshException(f"Expected non-empty 2D mesh array, got shape {mesh.shape}")

        # prefix area vector, if not already done (e.g. in STL format)
        if mesh.shape[1] == 3:
            row_number = int(len(content) / 3)
            mesh = mesh.reshape(row_number, 3, 3)
            mesh = np.hstack((Mesh._compute_face_normals(mesh[:, 0, :],mesh[:, 1, :],mesh[:, 2, :]), mesh))

        # Append addendum section - mesh is now in regular format and enums work appropriately
        face_count = mesh.shape[0]
        mesh = np.hstack((mesh, np.zeros((face_count, 2, 3))))

        # Inject Z values into A1
        mesh[:, M.A1, A1.V0Z] = mesh[:, M.V0, V.Z]
        mesh[:, M.A1, A1.V1Z] = mesh[:, M.V1, V.Z]
        mesh[:, M.A1, A1.V2Z] = mesh[:, M.V2, V.Z]

        mesh[:, M.A2, A2.MAX_Z] = np.max(mesh[:, M.V0:M.V2+1, V.Z], axis=1)
        mesh[:, M.A2, A2.MED_Z] = np.median(mesh[:, M.V0:M.V2+1, V.Z], axis=1)

        # Area is calculated as the magnitude of the cross product of two vectors, 
        # but halved (faces are triangles and not parallelograms)
        # See https://mathinsight.org/cross_product
        mesh[:, M.A2, A2.AREA] = np.sqrt(np.sum(np.square(mesh[:, M.NORM, :]), axis=-1)).reshape(face_count)
        mesh = mesh[mesh[:, M.A2, A2.AREA] != 0] # Filter faces without area
        face_count = mesh.shape[0]

        # Normalize the norm vector while we have its magnitude
        mesh[:, M.NORM, :] = mesh[:, M.NORM, :] / mesh[:, M.A2, A2.AREA].reshape(face_count, 1)

        mesh[:, M.A2, A2.AREA] = mesh[:, M.A2, A2.AREA] / 2  

        # remove small facets (essential for contour calculation)
        if negl_size is not None:
            filtered_mesh = mesh[np.where(mesh[:, M.A2, A2.AREA] > negl_size)]
            if len(filtered_mesh) > 100:
                mesh = filtered_mesh

        self.mesh = mesh

    def favour_side(self, orient: Orientation):
        """This function weights the size of faces closer than 45 deg
        to a favoured side higher, by inflating their computed area.
        Args:
            orient (Orientation): the favoured side e.g. [[0,1,0],3]
                                  Note that Orientation is always normalized
        Returns:
            a weighted mesh or the original mesh in case of invalid input
        """
        side = orient.vec
        COS_4385 = 0.721156 # cos(43.85 deg)
        print(f"You favour the side {side} with a factor of {orient.weight}")
        aligned = np.sum(self.mesh[:, M.NORM, :] * orient.vec, axis=1) > 0.7654  
        self.mesh[aligned, M.A2, A2.AREA] *= orient.weight

    def area_cumulation(self, best_n):
        """
        Gathering promising alignments by the accumulation of
        the magnitude of parallel area vectors.
        Args:
            best_n (int): amount of orientations to return.
        Returns:
            list of the common orientation-tuples.
        """
        orient = Counter()
        for i in range(len(self.mesh)):  # Accumulate area-vectors
            orient[tuple(self.mesh[i, M.NORM] + 0.0)] += self.mesh[i, M.A2, A2.AREA]

        return [Orientation(*tn) for tn in orient.most_common(best_n)]

    def death_star(self, best_n):
        """
        Creating random faces by adding a random vertex to an existing edge.
        Common orientations of these faces are promising orientations for
        placement.
        Args:
            best_n (int): amount of orientations to return.
        Returns:
            list of the common Orientations.
        """

        # Small files need more calculations
        # These values were probably picked as a best guess...
        mesh_len = len(self.mesh)
        iterations = int(np.ceil(20000 / (mesh_len + 100)))

        vertices = self.mesh[:, M.V0:M.V2+1, :]
        tot_normalized_orientations = np.zeros((iterations * mesh_len + 1, 3))
        # TODO use threadpool?
        for i in range(iterations):
            # Per face, we sample two of the three vertices at random
            sample = vertices[:, np.random.choice(3, 2, replace=False), :]
            v0 = sample[:, 0, :]
            v1 = sample[:, 1, :]

            # Select pseudorandom vertex
            # `i` is added to randomize between iterations.
            # See https://en.wikipedia.org/wiki/Linear_congruential_generator
            v2 = vertices[(np.arange(mesh_len) * Mesh.LIN_CONG_MULT + Mesh.LIN_CONG_INCR + i) % mesh_len, i % 3, :]
            normals = np.cross(np.subtract(v2, v0),
                               np.subtract(v1, v0))

            # Compute magnitudes of each normal to get parallelogram area
            lengths = np.sqrt((normals * normals).sum(axis=1)).reshape(mesh_len, 1)
            # Normalize the normal vectors, ignoring ZeroDivisions
            # and rounding to 6 decimal places 
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_orientations = np.around(np.true_divide(normals, lengths),
                                                    decimals=6)

            # Push orientations onto total
            tot_normalized_orientations[mesh_len * i:mesh_len * (i + 1)] = normalized_orientations
            sleep(0)  # Yield, so other threads get a bit of breathing space.

        # Accumulate the most common orientations by hashing the normal vectors.
        # TODO: we're hashing by three digits rather than 6 decimals as above...
        # unsure if this is totally correct.
        orient = Counter(np.inner(np.array([1, 1e3, 1e6]), tot_normalized_orientations))
        top_n = orient.most_common(best_n)

        # TODO: Why filter values to only those greater than 2? 
        # Doesn't this cut off a bunch of negative values?
        top_n = list(filter(lambda x: x[1] > 2, top_n))

        candidates = list()
        # For each top-ranking normal, count the number of unique faces of
        # that orientation and collect them as Orientations using the face count
        # as the weight.
        for sum_side, count in top_n:
            face_unique, face_count = np.unique(tot_normalized_orientations[orientations == sum_side], axis=0, return_counts=True)
            candidates += [Orientation(f, c) for f, c in zip(face_unique, face_count)]
        # Filter non-injective singles, i.e. candidate orientations
        # where there was only a single face and the face we randomly generated.
        candidates = list(filter(lambda x: x[1] >= 2, candidates))
        # also add anti-parallel orientations (facing opposite direction)
        # as we don't know which side of the face is best "up"
        candidates += [Orientation(-c.vec, c.weight) for c in candidates]
        return candidates

    def project_vertices(self, orientation):
        """Supplement the mesh array with scalars (max and median)
        for each face projected onto the orientation vector.
        Args:
            orientation (np.array): with format 3 x 3.
        Returns:
            None
        """
        self.mesh[:, M.A1, A1.V0Z] = np.inner(self.mesh[:, V0, :], orientation)
        self.mesh[:, M.A1, A1.V1Z] = np.inner(self.mesh[:, V1, :], orientation)
        self.mesh[:, M.A1, A1.V2Z] = np.inner(self.mesh[:, V2, :], orientation)

        self.mesh[:, M.A2, A2.MAX_Z] = np.max(self.mesh[:, M.A1, :], axis=1)
        self.mesh[:, M.A2, A2.MED_Z] = np.median(self.mesh[:, M.A1, :], axis=1)

    def calc_overhang(self, orientation, min_volume):
        """Calculating bottom and overhang area for a mesh regarding
        the vector n.
        Args:
            orientation (np.array): with format 3 x 3.
            min_volume (bool): minimize the support material volume or supported surfaces
        Returns:
            the total bottom size, overhang size and contour length of the mesh
        """
        total_min = np.amin(self.mesh[:, M.A1, :])

        # filter bottom area
        bottom = np.sum(self.mesh[np.where(self.mesh[:, M.A2, MA2.MAX_Z] < total_min + self.FIRST_LAY_H), M.A2, A2.AREA])
        # # equal than:
        # bottoms = mesh[np.where(mesh[:, 5, 1] < total_min + FIRST_LAY_H)]
        # if len(bottoms) > 0: bottom = np.sum(bottoms[:, 5, 0])
        # else: bottom = 0

        # filter overhangs
        overhangs = self.mesh[np.where(np.inner(self.mesh[:, M.NORM, :], orientation) < self.ASCENT)]
        overhangs = overhangs[np.where(overhangs[:, M.A2, A2.MAX_Z] > (total_min + self.FIRST_LAY_H))]

        if self.extended_mode:
            plafond = np.sum(overhangs[(overhangs[:, M.NORM, :] == -orientation).all(axis=1), M.A2, A2.AREA])
        else:
            plafond = 0

        if len(overhangs) > 0:
            if min_volume:
                heights = np.inner(overhangs[:, M.V0:M.V2+1, :].mean(axis=1), orientation) - total_min

                inner = np.inner(overhangs[:, M.NORM, :], orientation) - self.ASCENT
                # overhang = np.sum(heights * overhangs[:, 5, 0] * np.abs(inner * (inner < 0)) ** 2)
                overhang = np.sum((self.height_offset + self.height_log * np.log(self.height_log_k * heights + 1)) *
                                  overhangs[:, M.A2, A2.AREA] * np.abs(inner * (inner < 0)) ** self.OV_H)
            else:
                # overhang = np.sum(overhangs[:, 5, 0] * 2 *
                #                   (np.amax((np.zeros(len(overhangs)) + 0.5,
                #                             - np.inner(overhangs[:, 0, :], orientation)),
                #                            axis=0) - 0.5) ** 2)
                # improved performance by finding maximum using the multiplication method, see:
                # https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
                inner = np.inner(overhangs[:, M.NORM, :], orientation) - self.ASCENT
                overhang = 2 * np.sum(overhangs[:, M.A2, A2.AREA] * np.abs(inner * (inner < 0)) ** 2)
            overhang -= self.PLAFOND_ADV * plafond

        else:
            overhang = 0

        # filter the total length of the bottom area's contour
        if self.extended_mode:
            # contours = self.mesh[total_min+self.FIRST_LAY_H < self.mesh[:, 5, 1]]
            contours = self.mesh[np.where(self.mesh[:, M.A2, A2.MED_Z] < total_min + self.FIRST_LAY_H)]

            if len(contours) > 0:
                conlen = np.arange(len(contours))
                sortsc0 = np.argsort(contours[:, M.A1, :], axis=1)[:, 0]
                sortsc1 = np.argsort(contours[:, M.A1, :], axis=1)[:, 1]

                con = np.array([np.subtract(
                    contours[conlen, 1 + sortsc0, :],
                    contours[conlen, 1 + sortsc1, :])])

                contours = np.sum(np.power(con, 2), axis=-1) ** 0.5
                contour = np.sum(contours) + self.CONTOUR_AMOUNT * len(contours)
            else:
                contour = 0
        else:  # consider the bottom area as square, bottom=a**2 ^ contour=4*a
            contour = 4 * np.sqrt(bottom)
        return bottom, overhang, contour

    def euler(self, bestside):
        """Calculating euler rotation parameters and rotational matrix.
        Args:
            bestside (np.array): vector of the best orientation (3).
        Returns:
            rotation axis, rotation angle, rotational matrix.
        """
        if not isinstance(bestside, (list, np.ndarray)) or len(bestside) != 3:
            print(f"Best side not as excepted: {bestside}, type: {type(bestside)}")
        if bestside[0] ** 2 + bestside[1] ** 2 + (bestside[2] + 1.) ** 2 < abs(self.VECTOR_TOL):
            rotation_axis = [1., 0., 0.]
            phi = np.pi
        elif bestside[0] ** 2 + bestside[1] ** 2 + (bestside[2] - 1.) ** 2 < abs(self.VECTOR_TOL):
            rotation_axis = [1., 0., 0.]
            phi = 0.
        else:
            phi = np.pi - np.arccos(-bestside[2] + 0.0)
            rotation_axis = np.array(
                [-bestside[1] + 0.0, bestside[0] + 0.0, 0.])  # the z-axis is fixed to 0 for this rotation
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # normalization

        v = rotation_axis
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        rotational_matrix = np.empty((3, 3), dtype=np.float64)
        rotational_matrix[0, 0] = v[0] * v[0] * (1 - cos_phi) + cos_phi
        rotational_matrix[0, 1] = v[0] * v[1] * (1 - cos_phi) - v[2] * sin_phi
        rotational_matrix[0, 2] = v[0] * v[2] * (1 - cos_phi) + v[1] * sin_phi
        rotational_matrix[1, 0] = v[1] * v[0] * (1 - cos_phi) + v[2] * sin_phi
        rotational_matrix[1, 1] = v[1] * v[1] * (1 - cos_phi) + cos_phi
        rotational_matrix[1, 2] = v[1] * v[2] * (1 - cos_phi) - v[0] * sin_phi
        rotational_matrix[2, 0] = v[2] * v[0] * (1 - cos_phi) - v[1] * sin_phi
        rotational_matrix[2, 1] = v[2] * v[1] * (1 - cos_phi) + v[0] * sin_phi
        rotational_matrix[2, 2] = v[2] * v[2] * (1 - cos_phi) + cos_phi

        return [list(rotation_axis), phi, rotational_matrix]
