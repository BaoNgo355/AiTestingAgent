"""Module xử lý lịch biểu, ngày tháng và khung giờ làm việc phục vụ thực nghiệm kiểm thử (Độ phức tạp: Trung bình)."""

from __future__ import annotations

import datetime


def is_leap_year(year: int) -> bool:
    """Kiểm tra năm nhuận theo lịch Григорий (Gregorian)."""
    if year < 1:
        raise ValueError("year must be a positive integer (>= 1)")
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0


def days_in_month(year: int, month: int) -> int:
    """Trả về số ngày của một tháng trong năm."""
    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12")
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    return 29 if is_leap_year(year) else 28


def is_business_day(date_obj: datetime.date, holidays: set[datetime.date] | None = None) -> bool:
    """Kiểm tra một ngày có phải ngày làm việc (Thứ Hai..Thứ Sáu và không thuộc ngày lễ) không."""
    if date_obj.weekday() >= 5:
        return False
    if holidays and date_obj in holidays:
        return False
    return True


def is_within_working_hours(
    dt: datetime.datetime,
    *,
    start_hour: int = 8,
    end_hour: int = 17,
    holidays: set[datetime.date] | None = None,
) -> bool:
    """Kiểm tra thời điểm `dt` có nằm trong khung giờ làm việc [start_hour:00, end_hour:00] của ngày làm việc không."""
    if start_hour < 0 or end_hour > 24 or start_hour >= end_hour:
        raise ValueError("Invalid working hour range")
    if not is_business_day(dt.date(), holidays):
        return False
    total_minutes = dt.hour * 60 + dt.minute
    start_minutes = start_hour * 60
    end_minutes = end_hour * 60
    return start_minutes <= total_minutes <= end_minutes


def next_business_day(
    date_obj: datetime.date,
    *,
    days_ahead: int = 1,
    holidays: set[datetime.date] | None = None,
) -> datetime.date:
    """Tìm ngày làm việc tiếp theo sau `days_ahead` ngày làm việc."""
    if days_ahead < 0:
        raise ValueError("days_ahead must be non-negative")
    current = date_obj
    if days_ahead == 0:
        while not is_business_day(current, holidays):
            current += datetime.timedelta(days=1)
        return current

    remaining = days_ahead
    while remaining > 0:
        current += datetime.timedelta(days=1)
        if is_business_day(current, holidays):
            remaining -= 1
    return current


def has_schedule_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> bool:
    """Kiểm tra hai khoảng thời gian [start_a, end_a) và [start_b, end_b) có giao nhau không.

    Lưu ý: Nếu một sự kiện kết thúc đúng lúc sự kiện kia bắt đầu (end_a == start_b), không coi là trùng lịch.
    """
    if start_a >= end_a or start_b >= end_b:
        raise ValueError("Start time must be strictly less than end time")
    return start_a < end_b and start_b < end_a


def calculate_parking_fee(hours: float, *, is_weekend: bool = False, vip_member: bool = False) -> float:
    """Tính phí gửi xe theo bậc thang số giờ."""
    if hours < 0:
        raise ValueError("hours must be non-negative")
    if hours == 0:
        return 0.0
    if hours <= 1.0:
        base_fee = 10.0
    elif hours <= 4.0:
        base_fee = 10.0 + (hours - 1.0) * 8.0
    else:
        base_fee = 34.0 + (hours - 4.0) * 5.0

    if is_weekend:
        base_fee *= 1.2
    if vip_member:
        base_fee *= 0.85
    return round(base_fee, 2)
