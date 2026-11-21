"""Module quản lý tài khoản ngân hàng, hạn mức giao dịch và tính lãi suất (Độ phức tạp: Cao)."""

from __future__ import annotations


class AccountFrozenError(RuntimeError):
    """Ngoại lệ khi thực hiện giao dịch trên tài khoản đang bị khóa."""


class InsufficientFundsError(ValueError):
    """Ngoại lệ khi số dư khả dụng không đủ để rút hoặc chuyển khoản."""


class DailyLimitExceededError(ValueError):
    """Ngoại lệ khi vượt quá hạn mức rút/chuyển khoản trong ngày."""


class BankAccount:
    """Quản lý trạng thái tài khoản ngân hàng, nạp/rút tiền, chuyển khoản và tính lãi."""

    def __init__(
        self,
        account_id: str,
        owner: str,
        initial_balance: float = 0.0,
        *,
        daily_limit: float = 5000.0,
        overdraft_limit: float = 0.0,
    ) -> None:
        if not account_id or not account_id.strip():
            raise ValueError("account_id must not be empty")
        if not owner or not owner.strip():
            raise ValueError("owner must not be empty")
        if initial_balance < 0:
            raise ValueError("initial_balance must be non-negative")
        if daily_limit <= 0:
            raise ValueError("daily_limit must be strictly positive")
        if overdraft_limit < 0:
            raise ValueError("overdraft_limit must be non-negative")

        self.account_id = account_id.strip()
        self.owner = owner.strip()
        self._balance = float(initial_balance)
        self.daily_limit = float(daily_limit)
        self.overdraft_limit = float(overdraft_limit)
        self._daily_withdrawn = 0.0
        self._is_frozen = False
        self._transaction_count = 0

    @property
    def balance(self) -> float:
        return round(self._balance, 2)

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def daily_withdrawn(self) -> float:
        return round(self._daily_withdrawn, 2)

    @property
    def transaction_count(self) -> int:
        return self._transaction_count

    def freeze(self) -> None:
        """Khóa tài khoản tạm thời."""
        self._is_frozen = True

    def unfreeze(self) -> None:
        """Mở khóa tài khoản."""
        self._is_frozen = False

    def reset_daily_usage(self) -> None:
        """Đặt lại bộ đếm hạn mức rút tiền hằng ngày về 0."""
        self._daily_withdrawn = 0.0

    def deposit(self, amount: float) -> float:
        """Nạp tiền vào tài khoản; trả về số dư mới."""
        if self._is_frozen:
            raise AccountFrozenError(f"Account '{self.account_id}' is frozen")
        if amount <= 0:
            raise ValueError("Deposit amount must be strictly positive")
        self._balance += amount
        self._transaction_count += 1
        return self.balance

    def withdraw(self, amount: float) -> float:
        """Rút tiền từ tài khoản có kiểm tra khóa, số dư + thấu chi và hạn mức ngày."""
        if self._is_frozen:
            raise AccountFrozenError(f"Account '{self.account_id}' is frozen")
        if amount <= 0:
            raise ValueError("Withdrawal amount must be strictly positive")
        if self._daily_withdrawn + amount > self.daily_limit:
            raise DailyLimitExceededError(
                f"Withdrawal of {amount} exceeds daily limit {self.daily_limit} "
                f"(already used {self._daily_withdrawn})"
            )
        available = self._balance + self.overdraft_limit
        if amount > available:
            raise InsufficientFundsError(
                f"Insufficient funds: requested {amount}, available {available}"
            )
        self._balance -= amount
        self._daily_withdrawn += amount
        self._transaction_count += 1
        return self.balance

    def transfer_to(self, target: BankAccount, amount: float) -> tuple[float, float]:
        """Chuyển khoản sang tài khoản khác; đảm bảo tính nguyên tử nếu tài khoản đích bị khóa."""
        if not isinstance(target, BankAccount):
            raise TypeError("target must be a BankAccount instance")
        if target.account_id == self.account_id:
            raise ValueError("Cannot transfer to the same account")
        if self._is_frozen:
            raise AccountFrozenError(f"Source account '{self.account_id}' is frozen")
        if target.is_frozen:
            raise AccountFrozenError(f"Target account '{target.account_id}' is frozen")

        self.withdraw(amount)
        target.deposit(amount)
        return self.balance, target.balance

    def calculate_annual_interest(self) -> float:
        """Tính tiền lãi năm theo biểu lãi suất bậc thang trên số dư dương."""
        if self._balance <= 0:
            return 0.0
        if self._balance < 1000.0:
            rate = 0.01
        elif self._balance < 10000.0:
            rate = 0.025
        else:
            rate = 0.04
        return round(self._balance * rate, 2)
