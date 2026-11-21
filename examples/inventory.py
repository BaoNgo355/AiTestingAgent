"""Module quản lý kho hàng, chiết khấu bậc thang và cảnh báo đặt hàng lại (Độ phức tạp: Cao)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InventoryItem:
    """Thông tin một mặt hàng trong kho."""

    sku: str
    name: str
    unit_price: float
    quantity: int = 0
    reorder_threshold: int = 10

    def __post_init__(self) -> None:
        if not self.sku or not self.sku.strip():
            raise ValueError("SKU must not be empty")
        if self.unit_price < 0:
            raise ValueError("unit_price must be non-negative")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if self.reorder_threshold < 0:
            raise ValueError("reorder_threshold must be non-negative")


def calculate_tiered_discount(subtotal: float, *, customer_tier: str = "standard") -> float:
    """Tính tỷ lệ chiết khấu (0.0 .. 0.30) theo tổng giá trị đơn hàng và hạng khách hàng."""
    if subtotal < 0:
        raise ValueError("subtotal must be non-negative")
    tier = customer_tier.strip().lower()
    if tier not in ("standard", "silver", "gold"):
        raise ValueError(f"Unsupported customer_tier: {customer_tier}")

    if subtotal >= 1000.0:
        rate = 0.15
    elif subtotal >= 500.0:
        rate = 0.10
    elif subtotal >= 100.0:
        rate = 0.05
    else:
        rate = 0.0

    if tier == "silver":
        rate += 0.03
    elif tier == "gold":
        rate += 0.07

    return round(min(rate, 0.25), 4)


class InventoryManager:
    """Quản lý danh mục mặt hàng, nhập/xuất kho và tính giá trị đơn hàng."""

    def __init__(self) -> None:
        self._items: dict[str, InventoryItem] = {}

    def add_item(
        self,
        sku: str,
        name: str,
        unit_price: float,
        quantity: int = 0,
        reorder_threshold: int = 10,
    ) -> InventoryItem:
        """Thêm mặt hàng mới vào kho; ném ValueError nếu mã SKU đã tồn tại."""
        clean_sku = sku.strip().upper()
        if clean_sku in self._items:
            raise ValueError(f"Item with SKU '{clean_sku}' already exists")
        item = InventoryItem(
            sku=clean_sku,
            name=name.strip(),
            unit_price=unit_price,
            quantity=quantity,
            reorder_threshold=reorder_threshold,
        )
        self._items[clean_sku] = item
        return item

    def get_item(self, sku: str) -> InventoryItem:
        """Truy vấn mặt hàng theo SKU; ném KeyError nếu không tồn tại."""
        clean_sku = sku.strip().upper()
        if clean_sku not in self._items:
            raise KeyError(f"SKU '{clean_sku}' not found")
        return self._items[clean_sku]

    def restock(self, sku: str, amount: int) -> int:
        """Nhập thêm hàng vào kho; trả về số lượng tồn kho mới."""
        if amount <= 0:
            raise ValueError("Restock amount must be strictly positive")
        item = self.get_item(sku)
        item.quantity += amount
        return item.quantity

    def reserve_stock(self, sku: str, amount: int) -> int:
        """Xuất/giữ hàng từ kho; ném ValueError nếu số lượng tồn không đủ."""
        if amount <= 0:
            raise ValueError("Reserve amount must be strictly positive")
        item = self.get_item(sku)
        if amount > item.quantity:
            raise ValueError(
                f"Insufficient stock for SKU '{item.sku}': requested {amount}, available {item.quantity}"
            )
        item.quantity -= amount
        return item.quantity

    def needs_reorder(self, sku: str) -> bool:
        """Kiểm tra mặt hàng có cần đặt hàng lại không (tồn kho <= reorder_threshold)."""
        item = self.get_item(sku)
        return item.quantity <= item.reorder_threshold

    def list_low_stock_skus(self) -> list[str]:
        """Trả về danh sách các SKU có lượng tồn kho thấp (đã sắp xếp tăng dần)."""
        return sorted(
            sku for sku, item in self._items.items() if item.quantity <= item.reorder_threshold
        )

    def calculate_order_total(
        self,
        order_lines: dict[str, int],
        *,
        customer_tier: str = "standard",
        tax_rate: float = 0.08,
    ) -> float:
        """Tính tổng tiền thanh toán của đơn hàng sau khi trừ chiết khấu và cộng thuế."""
        if not order_lines:
            raise ValueError("order_lines must not be empty")
        if tax_rate < 0 or tax_rate > 1.0:
            raise ValueError("tax_rate must be between 0.0 and 1.0")

        subtotal = 0.0
        for sku, qty in order_lines.items():
            if qty <= 0:
                raise ValueError(f"Order quantity for '{sku}' must be strictly positive")
            item = self.get_item(sku)
            if qty > item.quantity:
                raise ValueError(f"Insufficient stock for '{item.sku}'")
            subtotal += item.unit_price * qty

        discount_rate = calculate_tiered_discount(subtotal, customer_tier=customer_tier)
        discounted = subtotal * (1.0 - discount_rate)
        total = discounted * (1.0 + tax_rate)
        return round(total, 2)
