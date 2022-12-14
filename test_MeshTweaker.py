import pytest
import numpy as np

# import Tweaker modules
import FileHandler
from MeshTweaker import Tweak, Orientation

@pytest.mark.skip("TODO")
def test_init():
    pass

@pytest.mark.skip("TODO")
def test_target_function():
    pass

@pytest.mark.skip("TODO")
def test_preprocess():
    pass

@pytest.mark.skip("TODO")
def test_favour_side():
    pass

@pytest.mark.skip("TODO")
def test_area_cumulation():
    pass

@pytest.mark.skip("TODO")
def test_death_star():
    pass

def test_add_supplements():
    sup = Tweak.add_supplements()
    assert len(sup) > 0
    for s in sup:
        assert len(s.vec) == 3
        assert s.weight == 0.0

def test_remove_duplicates_all_unique():
    orients = Tweak.add_supplements()
    assert Tweak.remove_duplicates(orients) == orients

def test_remove_duplicates_empty():
    assert len(Tweak.remove_duplicates([])) == 0

def test_remove_duplicates_over_under():
    eps = 0.0001
    thresh = 5*np.pi/180 # Declared in remove_duplicates()
    o1 = Orientation(np.array([np.cos(0), np.sin(0), 0]))
    o2 = Orientation(np.array([np.cos(thresh-eps), np.sin(thresh-eps), 0]))
    o3 = Orientation(np.array([np.cos(thresh+eps), np.sin(thresh+eps), 0]))

    # Close enough angle to remove
    assert(Tweak.remove_duplicates([o1, o2]) == [o1])

    # Slightly over removal threshold
    assert(Tweak.remove_duplicates([o1, o3]) == [o1, o3])

@pytest.mark.skip("TODO")
def test_project_vertices():
    pass

@pytest.mark.skip("TODO")
def test_calc_overhang():
    pass

@pytest.mark.skip("TODO")
def test_update_progress():
    pass

@pytest.mark.skip("TODO")
def test_euler():
    pass

@pytest.mark.skip("TODO")
def test_str():
    pass

