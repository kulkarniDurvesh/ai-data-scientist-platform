"""Synthetic retail dataset unrelated to any real project data."""

import numpy as np
import pandas as pd


def retail_dataframe(rows: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regions = np.array(["North", "South", "East", "West"])
    categories = np.array(["Furniture", "Office Supplies", "Technology"])

    region = rng.choice(regions, rows)
    category = rng.choice(categories, rows)
    quantity = rng.integers(1, 15, rows)
    unit_price = rng.gamma(2.0, 40.0, rows).round(2)
    discount = rng.choice([0, 0.1, 0.2, 0.3], rows)

    # Technology sells at a higher price, so groups differ.
    unit_price = np.where(category == "Technology", unit_price * 3, unit_price)

    sales = (quantity * unit_price * (1 - discount)).round(2)
    profit = (sales * rng.normal(0.15, 0.08, rows)).round(2)

    order_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730, rows), unit="D"
    )

    customers = [f"CUST-{i:04d}" for i in rng.integers(0, 300, rows)]

    return pd.DataFrame({
        "OrderID": [f"ORD-{i:06d}" for i in range(rows)],
        "order_date": order_date,
        "customer_id": customers,
        "customer_name": [f"Customer {c[-4:]}" for c in customers],
        "Region": region,
        "Category": category,
        "Quantity": quantity,
        "UnitPrice": unit_price,
        "Discount": discount,
        "Sales": sales,
        "Profit": profit,
        "Returned": rng.choice(["Yes", "No"], rows, p=[0.08, 0.92]),
        "Comment": [
            "Customer asked for faster delivery and a follow-up call next week"
        ] * rows,
        "Phone": rng.integers(9_000_000_000, 9_999_999_999, rows),
        "days_to_ship": rng.integers(-3, 10, rows),
    })
