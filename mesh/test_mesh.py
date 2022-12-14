import pytest
import numpy as np
from math import cos, sin

from .orientation import Orientation
from .mesh import Mesh, InvalidMeshException, M, A1, A2

def test_init_invalid():
    with pytest.raises(InvalidMeshException):
        m = Mesh(None)

def test_init_empty():
    with pytest.raises(InvalidMeshException):
        m = Mesh([[]])

def test_init_single_faceup_vertarray():
    m = Mesh([[1, 1, 0], [-1, 1, 0], [0, -1, 0]])

    assert np.array_equal(m.mesh[0, M.NORM, :], [0,0,1])
    assert np.array_equal(m.mesh[0, M.A1, :], [0,0,0])
    assert np.array_equal(m.mesh[0, M.A2, :], [2, 0, 0])

def test_init_vertarray():
    m = Mesh([
        [0, 0, 0], [0, 1, 0], [0, 0, 1], # facing X
        [0, 0, 0], [-1, 0, 0], [0, 0, -1], # facing -Y
    ])

    # Normals
    assert np.array_equal(m.mesh[:, M.NORM, :], [[1,0,0], [0, -1, 0]])
    # z values
    assert np.array_equal(m.mesh[:, M.A1, :], [[0,0,1], [0, 0, -1]])
    # AREA, MAX_Z, MED_Z
    assert np.array_equal(m.mesh[:, M.A2, :], [[0.5, 1, 0], [0.5, 0, 0]])


def test_init_stlformat():
    # Like faceup_vertarray test, but with Z down explicitly declared
    # via first triplet normals
    # TODO STL normals are actually unit vectors per wikipedia - should
    # re-compute area vectors to ensure they are correctly sized.
    m = Mesh([
        # TODO change -4 to -1 after fixing
        [[0, 0, -4], [1, 1, 0], [-1, 1, 0], [0, -1, 0]],
    ])
    assert np.array_equal(m.mesh[0, M.NORM, :], [0,0,-1])
    assert np.array_equal(m.mesh[0, M.A1, :], [0,0,0])
    assert np.array_equal(m.mesh[0, M.A2, :], [2, 0, 0])

def test_init_remove_small_facets():
    bigface = [[0,0,0],[0,100,0],[0,0,100]]
    smallface = [[0,0,0],[0,1,0],[0,0,1]]
    # Large negl_size -> small faces removed
    m = Mesh(bigface * 101 + smallface * 10, negl_size = 10)
    assert m.mesh.shape[0] == 101

    # Small negl_size -> small faces kept
    m = Mesh(bigface * 101 + smallface * 10, negl_size = 0.01)
    assert m.mesh.shape[0] == 111

    # Large negl_size but only small faces -> kept
    m = Mesh(smallface * 110, negl_size = 10)
    assert m.mesh.shape[0] == 110

def test_favour_side():
    m = Mesh([
        [0,0,0],[0,1,0],[0,0,1], # Triangle with x-axis norm, area 0.5
        [0,0,0],[1,0,0],[0,1,0], # Triangle with z-axis norm, area 0.5
        [0,0,0],[1,0,0],[0, cos(np.pi/4), sin(np.pi/4)], # norm 45* off Z-up
        [0,0,0],[1,0,0],[0, cos(np.pi/6), sin(np.pi/6)], # norm 30* off Z-up
    ]) 
    m.favour_side(Orientation([0,0,1], 2))
    assert m.mesh[0, M.A2, A2.AREA] == 0.5
    assert m.mesh[1, M.A2, A2.AREA] == 1 # Z area doubled
    assert m.mesh[2, M.A2, A2.AREA] == 0.5 # 45* is too far, so not doubled
    assert m.mesh[3, M.A2, A2.AREA] == 1 # 30* is within range, Z area doubled

def test_area_cumulation():
    # Big area normal to Z vs small X norm
    m = Mesh([
        [0,0,0],[10,0,0],[0,10,0],
        [0,0,0],[0,1,0],[0,0,1]
    ])
    assert m.area_cumulation(best_n=1)[0] == Orientation([0,0,1], 50)

    # Multiple verts wins in cumulative area
    m = Mesh([
        [0,0,0],[1,0,0],[0,1,0],
        [0,0,0],[1,0,0],[0,1,0],
        [0,0,0],[0,1,0],[0,0,2]
    ])
    assert m.area_cumulation(best_n=1)[0] == Orientation([0,0,1], 1)

@pytest.mark.skip("TODO")
def test_death_star():
    pass

@pytest.mark.skip("TODO")
def test_project_vertices():
    pass

@pytest.mark.skip("TODO")
def test_calc_overhang():
    pass

@pytest.mark.skip("TODO")
def test_euler():
    pass

