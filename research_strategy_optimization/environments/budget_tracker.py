"""Explicit finite budget accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetTracker:
    initial: int
    remaining: int
    spent: int = 0

    @classmethod
    def create(cls, budget: int) -> "BudgetTracker":
        if budget < 0:
            raise ValueError("budget must be non-negative")
        return cls(initial=budget, remaining=budget)

    def consume(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        if amount > self.remaining:
            raise RuntimeError(f"budget exhausted: requested {amount}, remaining {self.remaining}")
        self.remaining -= amount
        self.spent += amount

    def refund(self, amount: int = 1) -> None:
        if amount < 0 or amount > self.spent:
            raise ValueError("invalid refund")
        self.remaining += amount
        self.spent -= amount

    def state(self) -> dict:
        return {"initial": self.initial, "remaining": self.remaining, "spent": self.spent}

