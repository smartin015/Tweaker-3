import pytest
import numpy as np

from .orientation import Orientation

def test_init_normalizes():
    assert Orientation([10,0,0],1) == Orientation([1,0,0], 1)
    assert Orientation([0,1,1], 5) == Orientation([0,5,5], 5)

def test_supplements():
    sup = Orientation.supplements()
    assert len(sup) == 18
    for i in range(len(sup)):
        for j in range(i+1, len(sup)):
            assert sup[i] != sup[j]

def test_decimate():
    # Same orientations get merged
    assert len(Orientation.decimate([
        Orientation([0,0,1]), 
        Orientation([0,0,1])])) == 1

    # Slighly different orientations are merged
    assert len(Orientation.decimate([
        Orientation([0,0,1]), 
        Orientation([0,1,100])])) == 1

    # Different orientations are retained
    assert len(Orientation.decimate([
        Orientation([0,0,1]), 
        Orientation([0,1,10])])) == 2

    # Opposite orientations are retained
    assert len(Orientation.decimate([
        Orientation([0,0,1]), 
        Orientation([0,0,-1])])) == 2

def test_euler():
    TOL = -0.0008385913582234466 # From MeshTweaker.py

    # Test with Z-up orientation
    r, phi, m = Orientation([0,0,1]).euler(TOL)
    assert np.allclose(r, [1,0,0])
    assert phi == 0
    assert np.allclose(m, [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    # Test with Z-down orientation
    r, phi, m = Orientation([0,0,-1]).euler(TOL)
    assert np.allclose(r, [1,0,0])
    assert phi == np.pi
    assert np.allclose(m, [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])

    # Test X axis orientation (i.e. rot 90 deg around the Y axis)
    r, phi, m = Orientation([1,0,0]).euler(TOL)
    assert np.allclose(r, [0,1,0])
    assert phi == np.pi/2
    assert np.allclose(m, [
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
    ])

