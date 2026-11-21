from calculator import add, subtract, multiply, divide, factorial, power, is_prime, weighted_average, solve_quadratic
import pytest

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 0) == 0
    assert subtract(-1, -1) == 0

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-1, 1) == -1
    assert multiply(0, 5) == 0

def test_divide():
    assert divide(10, 2) == 5
    assert divide(0, 1) == 0
    with pytest.raises(ValueError):
        divide(1, 0)

def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    with pytest.raises(TypeError):
        factorial(3.5)
    with pytest.raises(ValueError):
        factorial(-1)

def test_power():
    assert power(2, 3) == 8
    assert power(0, 5) == 0
    assert power(2, 0) == 1
    with pytest.raises(ValueError):
        power(0, -1)

def test_is_prime():
    assert is_prime(2) == True
    assert is_prime(3) == True
    assert is_prime(4) == False
    assert is_prime(1) == False
    assert is_prime(0) == False

def test_weighted_average():
    assert weighted_average([1, 2, 3], [1, 1, 1]) == 2.0
    assert weighted_average([1, 2, 3], [2, 1, 1]) == 2.5
    with pytest.raises(ValueError):
        weighted_average([], [1, 2])
    with pytest.raises(ValueError):
        weighted_average([1, 2], [])
    with pytest.raises(ValueError):
        weighted_average([1, 2], [-1, 1])
    with pytest.raises(ValueError):
        weighted_average([1, 2], [1, 0])

def test_solve_quadratic():
    assert solve_quadratic(1, -3, 2) == (1.0, 2.0)
    assert solve_quadratic(1, 2, 1) == (-1.0,)
    assert solve_quadratic(1, 0, 1) == ()
    with pytest.raises(ValueError):
        solve_quadratic(0, 1, 1)
