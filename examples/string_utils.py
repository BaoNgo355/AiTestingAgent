"""Module xử lý và kiểm tra chuỗi ký tự phục vụ thực nghiệm kiểm thử (Độ phức tạp: Thấp - Trung bình)."""

from __future__ import annotations

import re


def is_palindrome(text: str, *, ignore_case: bool = True, alphanumeric_only: bool = True) -> bool:
    """Kiểm tra chuỗi đối xứng (palindrome)."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    processed = text
    if alphanumeric_only:
        processed = "".join(ch for ch in processed if ch.isalnum())
    if ignore_case:
        processed = processed.lower()
    if len(processed) <= 1:
        return True
    return processed == processed[::-1]


def slugify(text: str, *, max_length: int = 64) -> str:
    """Chuyển đổi chuỗi tiêu đề thành URL slug chuẩn (chữ thường, phân cách bằng dấu gạch ngang)."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s_-]+", "", cleaned)
    cleaned = re.sub(r"[\s_-]+", "-", cleaned).strip("-")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("-")
    return cleaned


def validate_email(email: str) -> bool:
    """Kiểm tra định dạng địa chỉ email cơ bản."""
    if not isinstance(email, str):
        return False
    candidate = email.strip()
    if len(candidate) < 5 or len(candidate) > 254:
        return False
    if " " in candidate or candidate.count("@") != 1:
        return False
    local_part, domain_part = candidate.split("@")
    if not local_part or not domain_part:
        return False
    if len(local_part) > 64:
        return False
    if local_part.startswith(".") or local_part.endswith(".") or ".." in local_part:
        return False
    if "." not in domain_part or domain_part.startswith(".") or domain_part.endswith("."):
        return False
    if ".." in domain_part:
        return False
    tld = domain_part.rsplit(".", 1)[-1]
    if len(tld) < 2 or not tld.isalpha():
        return False
    return True


def truncate_words(text: str, max_words: int, *, suffix: str = "...") -> str:
    """Cắt ngắn chuỗi theo số lượng từ tối đa."""
    if max_words < 0:
        raise ValueError("max_words must be non-negative")
    if max_words == 0:
        return suffix if text.strip() else ""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + suffix


def caesar_cipher(text: str, shift: int) -> str:
    """Mã hóa/giải mã chuỗi chữ cái ASCII bằng mật mã Caesar."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized_shift = shift % 26
    if normalized_shift == 0 or not text:
        return text
    result_chars: list[str] = []
    for ch in text:
        if "a" <= ch <= "z":
            offset = (ord(ch) - ord("a") + normalized_shift) % 26
            result_chars.append(chr(ord("a") + offset))
        elif "A" <= ch <= "Z":
            offset = (ord(ch) - ord("A") + normalized_shift) % 26
            result_chars.append(chr(ord("A") + offset))
        else:
            result_chars.append(ch)
    return "".join(result_chars)


def password_strength_score(password: str) -> int:
    """Chấm điểm độ mạnh mật khẩu trên thang 0..5."""
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    if len(password) < 6:
        return 0
    score = 1
    if len(password) >= 12:
        score += 1
    if any(ch.islower() for ch in password) and any(ch.isupper() for ch in password):
        score += 1
    if any(ch.isdigit() for ch in password):
        score += 1
    if any(not ch.isalnum() for ch in password):
        score += 1
    return score
