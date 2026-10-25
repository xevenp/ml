from __future__ import annotations

import pandas as pd

from data_processor import DataValidationError, _rename_aliases, validate_dataset


def product_analytics(sales: pd.DataFrame, product_master: pd.DataFrame | None = None) -> pd.DataFrame:
    records = validate_dataset(sales, "Sales_History")
    if "revenue_php" in records:
        records["revenue"] = records["revenue_php"]
    elif "unit_price_php" in records:
        records["revenue"] = records["quantity_sold"] * records["unit_price_php"]
    else:
        raise DataValidationError("Product analysis needs `revenue_php` or both `quantity_sold` and `unit_price_php`.")
    if "gross_profit_php" in records:
        records["gross_profit"] = records["gross_profit_php"]
    elif {"unit_price_php", "unit_cost_php"} <= set(records.columns):
        records["gross_profit"] = records["quantity_sold"] * (
            records["unit_price_php"] - records["unit_cost_php"]
        )
    else:
        records["gross_profit"] = 0.0
    if "category" not in records:
        records["category"] = "Uncategorized"
    aggregations = {
        "quantity": ("quantity_sold", "sum"),
        "revenue": ("revenue", "sum"),
        "gross_profit": ("gross_profit", "sum"),
        "category": ("category", "last"),
    }
    if "unit_price_php" in records:
        aggregations["unit_price"] = ("unit_price_php", "last")
    if "unit_cost_php" in records:
        aggregations["unit_cost"] = ("unit_cost_php", "last")
    products = records.groupby("product_name", as_index=False).agg(**aggregations)
    if product_master is not None:
        master = _rename_aliases(validate_dataset(product_master, "Product_Master"))
        if "category" in master:
            category_lookup = master.drop_duplicates("product_name", keep="last").set_index("product_name")["category"]
            products["category"] = products["product_name"].map(category_lookup).fillna(products["category"])
        for source, destination in (("unit_price_php", "unit_price"), ("unit_cost_php", "unit_cost")):
            if source in master:
                lookup = master.drop_duplicates("product_name", keep="last").set_index("product_name")[source]
                if destination not in products:
                    products[destination] = 0.0
                products[destination] = products["product_name"].map(lookup).fillna(products[destination])
    for column in ("unit_price", "unit_cost"):
        if column not in products:
            products[column] = 0.0
    return products.sort_values(["revenue", "quantity"], ascending=False).reset_index(drop=True)


def inventory_analytics(inventory: pd.DataFrame, sales: pd.DataFrame | None = None) -> pd.DataFrame:
    records = validate_dataset(inventory, "Inventory_History")
    records = records.sort_values(["product_name", "date"])
    latest = records.groupby("product_name", as_index=False).tail(1).copy()
    if "category" not in latest:
        latest["category"] = "Uncategorized"
    if "reorder_level" not in latest:
        latest["reorder_level"] = 0
    if "stockout_flag" not in latest:
        latest["stockout_flag"] = 0
    if "quantity_sold" in records:
        demand = records.groupby("product_name", as_index=False).agg(
            avg_daily_demand=("quantity_sold", "mean"),
            stockout_days=("stockout_flag", "sum") if "stockout_flag" in records else ("quantity_sold", lambda _: 0),
        )
    else:
        demand = records[["product_name"]].drop_duplicates().assign(avg_daily_demand=0.0, stockout_days=0)
    if sales is not None:
        sales_records = validate_dataset(sales, "Sales_History")
        sales_daily = sales_records.groupby(["product_name", "date"], as_index=False)["quantity_sold"].sum()
        sales_demand = sales_daily.groupby("product_name")["quantity_sold"].mean()
        demand["avg_daily_demand"] = demand["product_name"].map(sales_demand).fillna(demand["avg_daily_demand"])
    result = latest.merge(demand, on="product_name", how="left", suffixes=("", "_history"))
    result["avg_daily_demand"] = result["avg_daily_demand"].fillna(0).clip(lower=0)
    result["stockout_days"] = result["stockout_days"].fillna(0).astype(int)
    result["reorder_level"] = result["reorder_level"].fillna(0)
    result["low_stock"] = result["closing_stock"] <= result["reorder_level"]
    target_stock = result["avg_daily_demand"] * 7
    result["suggested_restock"] = (target_stock - result["closing_stock"]).clip(lower=0).round().astype(int)
    columns = [
        "product_name", "category", "opening_stock", "closing_stock", "reorder_level",
        "avg_daily_demand", "stockout_days", "suggested_restock", "low_stock",
    ]
    return result[columns].sort_values(["low_stock", "suggested_restock"], ascending=False).reset_index(drop=True)


def business_insights(
    daily: pd.DataFrame,
    products: pd.DataFrame | None = None,
    inventory: pd.DataFrame | None = None,
) -> list[str]:
    insights: list[str] = []
    if not daily.empty:
        best = daily.loc[daily["revenue"].idxmax()]
        insights.append(
            f"Highest recorded revenue day was {pd.Timestamp(best['date']).strftime('%B %d, %Y')} "
            f"(₱{best['revenue']:,.2f})."
        )
        weekday_data = daily.assign(
            day_type=daily["date"].dt.day_of_week.map(lambda day: "Weekend" if day >= 5 else "Weekday")
        ).groupby("day_type")["revenue"].mean()
        if {"Weekday", "Weekend"} <= set(weekday_data.index):
            high = "weekends" if weekday_data["Weekend"] > weekday_data["Weekday"] else "weekdays"
            insights.append(
                f"Average daily revenue is higher on {high}: weekends ₱{weekday_data['Weekend']:,.2f} "
                f"vs weekdays ₱{weekday_data['Weekday']:,.2f}."
            )
    if products is not None and not products.empty:
        revenue_leader = products.sort_values("revenue", ascending=False).iloc[0]
        quantity_leader = products.sort_values("quantity", ascending=False).iloc[0]
        insights.append(
            f"{revenue_leader['product_name']} generated the highest product revenue "
            f"(₱{revenue_leader['revenue']:,.2f})."
        )
        insights.append(
            f"{quantity_leader['product_name']} sold the most units "
            f"({quantity_leader['quantity']:,.0f})."
        )
    if inventory is not None and not inventory.empty:
        low = inventory[inventory["low_stock"]]
        if low.empty:
            insights.append("No product is at or below its recorded reorder level in the latest stock records.")
        else:
            names = ", ".join(low["product_name"].head(5).astype(str))
            extra = f" and {len(low) - 5} more" if len(low) > 5 else ""
            insights.append(f"{len(low)} product(s) are at or below reorder level: {names}{extra}.")
    return insights
