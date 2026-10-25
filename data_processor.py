from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


DATASET_TYPES = ("Sales_History", "Daily_Summary", "Inventory_History", "Product_Master")

ALIASES = {
    "date": {"date", "sale_date", "transaction_date", "movement_date", "day"},
    "product_name": {"product", "product_name", "item", "item_name"},
    "category": {"category", "product_category"},
    "quantity_sold": {"quantity", "quantity_sold", "units_sold", "qty_sold"},
    "total_quantity_sold": {"total_quantity", "total_quantity_sold", "units_sold"},
    "total_revenue_php": {"revenue", "total_revenue", "total_revenue_php", "sales", "sales_revenue"},
    "revenue_php": {"revenue", "revenue_php", "sales_revenue"},
    "gross_profit_php": {"gross_profit", "gross_profit_php", "profit"},
    "unit_price_php": {"unit_price", "unit_price_php", "price"},
    "unit_cost_php": {"unit_cost", "unit_cost_php", "cost"},
    "opening_stock": {"opening_stock", "opening_quantity", "stock_opening"},
    "stock_received": {"stock_received", "received_stock", "quantity_received"},
    "closing_stock": {"closing_stock", "ending_stock", "stock_on_hand", "available_stock"},
    "reorder_level": {"reorder_level", "reorder_point", "minimum_stock"},
    "stockout_flag": {"stockout_flag", "stockout", "is_stockout"},
}

REQUIRED_COLUMNS = {
    "Sales_History": {"date", "product_name", "quantity_sold"},
    "Daily_Summary": {"date"},
    "Inventory_History": {"date", "product_name", "opening_stock", "closing_stock"},
    "Product_Master": {"product_name"},
}


class DataValidationError(ValueError):
    """Raised when an uploaded dataset cannot safely support its intended analysis."""


def _normalize_column(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _rename_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [_normalize_column(column) for column in frame.columns]
    existing = set(frame.columns)
    rename: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        if canonical in existing:
            continue
        match = next((alias for alias in aliases if alias in existing), None)
        if match is not None:
            rename[match] = canonical
            existing.add(canonical)
    return frame.rename(columns=rename)


def infer_dataset_type(frame: pd.DataFrame) -> str:
    columns = set(_rename_aliases(frame).columns)
    if {"date", "product_name", "opening_stock", "closing_stock"} <= columns:
        return "Inventory_History"
    if {"date", "product_name", "quantity_sold"} <= columns:
        return "Sales_History"
    if "date" in columns and ({"total_revenue_php", "revenue_php"} & columns or "total_quantity_sold" in columns):
        return "Daily_Summary"
    if "product_name" in columns and "unit_price_php" in columns:
        return "Product_Master"
    raise DataValidationError(
        "Could not identify this dataset. Include date and product/stock fields for "
        "inventory, date and revenue or quantity for daily sales, or product and price "
        "for the product catalog."
    )


def validate_dataset(frame: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    if dataset_type not in REQUIRED_COLUMNS:
        raise DataValidationError(f"Unsupported worksheet type: {dataset_type}.")
    result = _rename_aliases(frame)
    missing = REQUIRED_COLUMNS[dataset_type] - set(result.columns)
    if missing:
        raise DataValidationError(
            f"{dataset_type} is missing required column(s): {', '.join(sorted(missing))}. "
            f"Columns found: {', '.join(map(str, result.columns))}."
        )
    if result.empty:
        raise DataValidationError(f"{dataset_type} contains no data rows.")
    if dataset_type == "Daily_Summary" and not (
        {"total_revenue_php", "revenue_php", "total_quantity_sold", "quantity_sold"} & set(result.columns)
    ):
        raise DataValidationError(
            "Daily_Summary needs at least one numeric target column: revenue or quantity sold."
        )
    if "date" in result:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        if result["date"].isna().any():
            count = int(result["date"].isna().sum())
            raise DataValidationError(f"{count} row(s) have a missing or invalid date.")
        result["date"] = result["date"].dt.normalize()
    numeric_by_type = {
        "Sales_History": ["quantity_sold", "unit_price_php", "unit_cost_php", "revenue_php", "gross_profit_php"],
        "Daily_Summary": ["total_quantity_sold", "total_revenue_php", "gross_profit_php"],
        "Inventory_History": ["opening_stock", "stock_received", "quantity_sold", "closing_stock", "reorder_level", "stockout_flag"],
        "Product_Master": ["unit_price_php", "unit_cost_php"],
    }[dataset_type]
    for column in numeric_by_type:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
            if result[column].isna().any():
                count = int(result[column].isna().sum())
                raise DataValidationError(f"{count} row(s) have a missing or invalid numeric value in `{column}`.")
            if not np.isfinite(result[column].to_numpy(dtype=float)).all():
                raise DataValidationError(f"`{column}` contains non-finite numeric values.")
    if "product_name" in result:
        result["product_name"] = result["product_name"].astype("string").str.strip()
        if result["product_name"].isna().any() or result["product_name"].eq("").any():
            raise DataValidationError("Product names cannot be blank.")
    result = result.sort_values("date", kind="stable").reset_index(drop=True) if "date" in result else result.reset_index(drop=True)
    result.attrs["dataset_type"] = dataset_type
    return result


def load_dataset(source: str | Path | BinaryIO, sheet_name: str | None = None) -> pd.DataFrame:
    name = str(getattr(source, "name", source))
    if name.lower().endswith(".xlsx"):
        try:
            frame = pd.read_excel(source, sheet_name=sheet_name, engine="openpyxl")
        except (ValueError, OSError, ImportError) as exc:
            raise DataValidationError(f"Unable to read Excel worksheet `{sheet_name}`: {exc}") from exc
    elif name.lower().endswith(".csv"):
        try:
            frame = pd.read_csv(source)
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            raise DataValidationError(f"Unable to read CSV file: {exc}") from exc
    else:
        raise DataValidationError("Unsupported file type. Upload an .xlsx or .csv file.")
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("Select one worksheet at a time.")
    dataset_type = sheet_name if sheet_name in REQUIRED_COLUMNS else infer_dataset_type(frame)
    return validate_dataset(frame, dataset_type)


def read_upload(uploaded_file: BinaryIO, sheet_name: str | None = None) -> pd.DataFrame:
    uploaded_file.seek(0)
    return load_dataset(uploaded_file, sheet_name)


def prepare_daily_sales(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "Daily_Summary" in tables:
        raw = _rename_aliases(tables["Daily_Summary"])
        if "total_revenue_php" not in raw and "revenue_php" not in raw:
            revenue = pd.Series(0.0, index=raw.index)
        else:
            revenue = raw.get("total_revenue_php", raw.get("revenue_php"))
        if "total_quantity_sold" in raw:
            quantity = raw["total_quantity_sold"]
        elif "quantity_sold" in raw:
            quantity = raw["quantity_sold"]
        else:
            quantity = pd.Series(0.0, index=raw.index)
        result = pd.DataFrame({"date": raw["date"], "revenue": revenue, "quantity": quantity})
    elif "Sales_History" in tables:
        raw = _rename_aliases(tables["Sales_History"])
        if "revenue_php" in raw:
            revenue = raw["revenue_php"]
        elif {"quantity_sold", "unit_price_php"} <= set(raw.columns):
            revenue = raw["quantity_sold"] * raw["unit_price_php"]
        else:
            revenue = pd.Series(0.0, index=raw.index)
        result = pd.DataFrame({
            "date": raw["date"],
            "revenue": revenue,
            "quantity": raw["quantity_sold"],
        })
    else:
        raise DataValidationError("Forecasting requires a Daily_Summary or Sales_History dataset.")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ("revenue", "quantity"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["date", "revenue", "quantity"]].isna().any().any():
        raise DataValidationError("Daily sales data contains invalid dates, revenue, or quantities.")
    if (result[["revenue", "quantity"]] < 0).any().any():
        raise DataValidationError("Sales revenue and quantity cannot be negative.")
    result = result.groupby("date", as_index=False)[["revenue", "quantity"]].sum()
    result = result.sort_values("date").reset_index(drop=True)
    expected_dates = pd.date_range(result["date"].min(), result["date"].max(), freq="D")
    missing_dates = expected_dates.difference(pd.DatetimeIndex(result["date"]))
    if len(missing_dates):
        raise DataValidationError(
            f"Daily sales history is missing {len(missing_dates)} calendar date(s). "
            "Add confirmed zero-sales days (or the missing records) before forecasting so lag features are aligned."
        )
    if len(result) < 10:
        raise DataValidationError("At least 10 distinct daily observations are required for a chronological evaluation.")
    return result
