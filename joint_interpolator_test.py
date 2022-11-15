from joint_interpolator import JointInterpolator
from numpy import array

threshold = 1

def test_interpolate_shape():
    j1 = \
        [[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7]]

    j2 = \
        [[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5]]

    js = array([j1, j2])

    ipltr = JointInterpolator(window_size = 3)
    jsi = ipltr.interpolate(js)

    assert jsi.shape == (2, 5, 3)
    
def test_interpolate_on_complete_data():
    j1 = \
        [[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7]]

    j2 = \
        [[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5]]

    js = array([j1, j2])

    ipltr = JointInterpolator(window_size = 3)
    jsi = ipltr.interpolate(js)

    assert (jsi == js).all()

def test_interpolate_on_missing_data():
    j1 = \
        [[1, 2, 3],
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [4, 5, 6],
        [5, 6, 7],
        [5, 6, 7]]

    j1_anomalous = \
        [[1, 2, 3],
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 4],
        [10, 10, 5],
        [4, 5, 6],
        [4, 5, 6],
        [5, 6, 7],
        [5, 6, 7]]

    j2 = \
        [[1, 1, 1],
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [4, 4, 4],
        [5, 5, 5],
        [5, 5, 5]]

    j2_anomalous = \
        [[1, 1, 1],
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [3, 10, 3],
        [4, 4, 4],
        [4, 4, 4],
        [5, 5, 5],
        [5, 5, 5]]

    js = array([j1_anomalous, j2_anomalous])
    js_expected = array([j1, j2])

    ipltr = JointInterpolator(window_size = 5)
    jsi = ipltr.interpolate(js)

    assert (abs(jsi - js_expected) <= threshold).all()

def test_interpolate_large_window():
    j1 = \
        [[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [3, 3, 4],
        [3, 3, 3],
        [3, 3, 3],
        [2, 2, 3],
        [2, 2, 3],
        [1, 2, 3],
        [2, 2, 2],
        [2, 2, 2]]

    j1_anomalous = \
        [[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [3, 3, 4],
        [10, 10, 10],
        [10, 10, 10],
        [2, 2, 3],
        [2, 2, 2],
        [1, 2, 3],
        [2, 2, 2],
        [2, 2, 2]]

    js = array([j1_anomalous])
    js_expected = array([j1])

    ipltr = JointInterpolator(window_size = 7)
    jsi = ipltr.interpolate(js)

    assert (abs(jsi - js_expected) <= threshold).all()
