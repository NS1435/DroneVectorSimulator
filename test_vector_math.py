"""
MATH 215 - Drone Vector Simulator
Mathematical Verification Tests

This file tests the vector calculations used in the drone simulator.

Run with:
    python test_vector_math.py
"""

import numpy as np
from drone_simulator import VectorMath as V


# --------------------------------------------------
# 1.1 - VECTORS AND COMPONENTS
# --------------------------------------------------

def test_magnitude():
    # A vector <3, 4> should have a magnitude of 5.
    assert np.isclose(
        V.magnitude([3, 4]), 5
    )


def test_unit_vector():
    # Dividing <3, 4> by its magnitude gives <0.6, 0.8>.
    assert np.allclose(
        V.normalize([3, 4]),
        [0.6, 0.8]
    )


def test_unit_vector_of_zero_is_undefined():
    # A zero vector has no direction.
    assert V.normalize([0, 0]) is None


# --------------------------------------------------
# 1.2 - VECTOR OPERATIONS
# --------------------------------------------------

def test_addition():
    # <2, 1> + <-1, 3> = <1, 4>
    assert np.allclose(
        V.resultant([2, 1], [-1, 3]),
        [1, 4]
    )


def test_subtraction():
    # <6, 8> - <2, 3> = <4, 5>
    assert np.allclose(
        V.subtract([6, 8], [2, 3]),
        [4, 5]
    )


def test_scalar_multiplication():
    # Multiplying <2, 3> by 2 gives <4, 6>.
    assert np.allclose(
        V.scalar_multiply(2, [2, 3]),
        [4, 6]
    )


# --------------------------------------------------
# 1.4 - DOT PRODUCTS AND ANGLES
# --------------------------------------------------

def test_dot_product():
    # (2)(4) + (3)(-1) = 5
    assert np.isclose(
        V.dot_product([2, 3], [4, -1]),
        5
    )


def test_orthogonal_angle():
    # Perpendicular vectors form a 90-degree angle.
    assert np.isclose(
        V.angle([1, 0], [0, 1]),
        90
    )


def test_angle_with_zero_vector_is_undefined():
    # An angle cannot be calculated with a zero vector.
    assert V.angle([0, 0], [1, 0]) is None


# --------------------------------------------------
# 1.5 - VECTOR PROJECTIONS
# --------------------------------------------------

def test_projection():
    # Projecting <3, 4> onto the x-axis gives <3, 0>.
    assert np.allclose(
        V.projection([3, 4], [1, 0]),
        [3, 0]
    )


def test_projection_onto_zero_is_undefined():
    # Projection onto a zero vector is undefined.
    assert V.projection([3, 4], [0, 0]) is None


def test_components_sum_to_original():
    # Split <3, 4> into parallel and perpendicular components.
    parallel, perpendicular = V.components(
        [3, 4],
        [1, 0]
    )

    # Parallel component points along the x-axis.
    assert np.allclose(
        parallel, [3, 0]
    )

    # Perpendicular component points along the y-axis.
    assert np.allclose(
        perpendicular, [0, 4]
    )

    # Adding both components should reproduce the original vector.
    assert np.allclose(
        parallel + perpendicular,
        [3, 4]
    )

    # The parallel and perpendicular components must be orthogonal.
    # Their dot product should equal zero.
    assert np.isclose(
        V.dot_product(
            parallel,
            perpendicular
        ),
        0
    )


# --------------------------------------------------
# RUN TESTS
# --------------------------------------------------

if __name__ == "__main__":

    # Find all functions whose names begin with "test_".
    tests = [
        f
        for name, f in list(globals().items())
        if name.startswith("test_")
    ]

    # Run each test and print its result.
    for test in tests:
        test()
        print("PASS", test.__name__)

    # Display the total number of successful tests.
    print(
        f"\n{len(tests)} of {len(tests)} tests passed"
    )