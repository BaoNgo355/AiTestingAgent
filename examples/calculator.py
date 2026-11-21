"""Module các hàm toán học cơ bản và nâng cao phục vụ thực nghiệm kiểm thử (Độ phức tạp: Thấp)."""

from __future__ import annotations

import math


def add(a: float, b: float) -> float:
    """Cộng hai số thực."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Trừ hai số thực."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Nhân hai số thực."""
    return a * b


def divide(a: float, b: float) -> float:
    """Chia hai số thực, ném ValueError nếu số chia bằng 0."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def factorial(n: int) -> int:
    """Tính giai thừa n! cho số nguyên không âm."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def power(base: float, exponent: int) -> float:
    """Tính lũy thừa base^exponent với số mũ nguyên."""
    if not isinstance(exponent, int) or isinstance(exponent, bool):
        raise TypeError("exponent must be an integer")
    if base == 0 and exponent < 0:
        raise ValueError("Zero cannot be raised to a negative power")
    if exponent == 0:
        return 1.0
    abs_exp = -exponent if exponent < 0 else exponent
    result = 1.0
    for _ in range(abs_exp):
        result *= base
    if exponent < 0:
        return 1.0 / result
    return result


def is_prime(n: int) -> bool:
    """Kiểm tra số nguyên tố."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.isqrt(n))
    for factor in range(3, limit + 1, 2):
        if n % factor == 0:
            return False
    return True


def weighted_average(values: list[float], weights: list[float]) -> float:
    """Tính trung bình cộng có trọng số."""
    if not values or not weights:
        raise ValueError("values and weights must not be empty")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Sum of weights must be greater than zero")
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight


def solve_quadratic(a: float, b: float, c: float) -> tuple[float, ...]:
    """Giải phương trình bậc hai a*x^2 + b*x + c = 0 trên tập số thực."""
    if a == 0:
        raise ValueError("Coefficient 'a' must not be zero for a quadratic equation")
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return ()
    if discriminant == 0:
        root = -b / (2 * a)
        return (root,)
    sqrt_d = math.sqrt(discriminant)
    r1 = (-b - sqrt_d) / (2 * a)
    r2 = (-b + sqrt_d) / (2 * a)
    return (min(r1, r2), max(r1, r2))
