import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises

from pyvrp import (
    Client,
    Depot,
    PiecewiseLinearFunction,
    ProblemData,
    Route,
    VehicleType,
)

_INT_MAX = int(np.iinfo(np.int64).max)


def test_piecewise_linear_function_evaluates_segments():
    fn = PiecewiseLinearFunction([0, 5, 10, 20], [(1, 2), (11, 3), (26, 4)])

    assert_equal(fn(0), 1)
    assert_equal(fn(4), 9)
    assert_equal(fn(5), 11)
    assert_equal(fn(9), 23)
    assert_equal(fn(10), 26)
    assert_equal(fn(13), 38)
    assert_equal(fn(20), 66)


def test_piecewise_linear_function_raises_invalid_data():
    with assert_raises(ValueError):
        PiecewiseLinearFunction([], [])

    with assert_raises(ValueError):
        PiecewiseLinearFunction([0], [])

    with assert_raises(ValueError):
        PiecewiseLinearFunction([0, 1], [])

    with assert_raises(ValueError):
        PiecewiseLinearFunction([0, 1, 1], [(0, 0), (0, 0)])


def test_piecewise_linear_function_raises_for_out_of_domain_query():
    fn = PiecewiseLinearFunction([5, 10], [(1, 2)])

    with assert_raises(ValueError, match="x must be within function domain."):
        fn(4)

    with assert_raises(ValueError, match="x must be within function domain."):
        fn(11)

    assert_equal(fn(10), 11)


def test_piecewise_linear_function_exposes_breakpoints_and_segments():
    fn = PiecewiseLinearFunction([0, 5, 10], [(1, 2), (11, 3)])

    assert_equal(fn.breakpoints, [0, 5, 10])
    assert_equal(fn.segments, [(1, 2), (11, 3)])


def test_piecewise_linear_function_is_zero():
    assert PiecewiseLinearFunction().is_zero()
    assert PiecewiseLinearFunction([0, 1], [(0, 0)]).is_zero()
    assert not PiecewiseLinearFunction([0, 1], [(0, 1)]).is_zero()
    assert not PiecewiseLinearFunction([0, 1], [(1, 0)]).is_zero()


def test_piecewise_linear_function_equality():
    fn1 = PiecewiseLinearFunction([0, 5], [(1, 2)])
    fn2 = PiecewiseLinearFunction([0, 5], [(1, 2)])
    fn3 = PiecewiseLinearFunction([0, 5], [(1, 3)])
    fn4 = PiecewiseLinearFunction([0, 6], [(1, 2)])
    fn5 = PiecewiseLinearFunction([0, 5], [(2, 2)])

    assert fn1 == fn2
    assert fn1 != fn3
    assert fn1 != fn4
    assert fn1 != fn5


def test_piecewise_linear_function_raises_unsorted_breakpoints():
    with assert_raises(
        ValueError,
        match="breakpoints must be strictly increasing.",
    ):
        PiecewiseLinearFunction([0, 2, 1], [(1, 1), (2, 2)])


def test_vehicle_type_uses_provided_duration_cost_function():
    duration_fn = PiecewiseLinearFunction(
        [0, 5, _INT_MAX],
        [(0, 1), (5, 10)],
    )
    vehicle_type = VehicleType(duration_cost_function=duration_fn)

    assert_equal(vehicle_type.duration_cost_function, duration_fn)
    assert_equal(vehicle_type.duration_cost_slope, 1)


def test_vehicle_type_defaults_to_zero_duration_cost_function():
    vehicle_type = VehicleType()

    assert vehicle_type.duration_cost_function.is_zero()
    assert_equal(
        vehicle_type.duration_cost_function.breakpoints,
        [np.iinfo(np.int64).min, _INT_MAX],
    )
    assert_equal(vehicle_type.duration_cost_function.segments, [(0, 0)])


def test_route_duration_cost_matches_vehicle_duration_cost_function():
    duration_fn = PiecewiseLinearFunction([0, 5, _INT_MAX], [(0, 1), (5, 10)])

    data = ProblemData(
        clients=[Client(x=0, y=1)],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(duration_cost_function=duration_fn)],
        distance_matrices=[np.array([[0, 0], [0, 0]], dtype=np.int64)],
        duration_matrices=[np.array([[0, 4], [4, 0]], dtype=np.int64)],
    )

    route = Route(data, visits=[1], vehicle_type=0)
    duration = route.duration()

    assert_equal(duration, 8)  # 4 from depot to client and 4 back.
    assert_equal(route.duration_cost(), 35)  # 1 * 5 + 10 * (8 - 5)
    assert_equal(
        route.duration_cost(), data.vehicle_type(0).duration_cost_function(8)
    )


def test_vehicle_type_replace_updates_duration_cost_function():
    original = PiecewiseLinearFunction([0, _INT_MAX], [(0, 1)])
    updated = PiecewiseLinearFunction([0, 5, _INT_MAX], [(0, 1), (5, 10)])
    vehicle_type = VehicleType(duration_cost_function=original)

    replaced = vehicle_type.replace(duration_cost_function=updated)
    unchanged = vehicle_type.replace()

    assert_equal(replaced.duration_cost_function, updated)
    assert_equal(unchanged.duration_cost_function, original)
