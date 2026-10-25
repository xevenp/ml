from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    business_insights,
    inventory_analytics,
    product_analytics,
)
from data_processor import (
    DATASET_TYPES,
    load_dataset,
    prepare_daily_sales,
    read_upload,
)
from train_model import evaluate_model, forecast_future


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sales_data.csv"
SAMPLE_WORKBOOK = SAMPLE_DIR / "KaPaye_ML_Historical_Sales_Dataset.xlsx"
SAMPLE_FILES = {
    "Sales_History": SAMPLE_DIR / "kapaye_historical_sales.csv",
    "Daily_Summary": SAMPLE_DIR / "kapaye_daily_sales_summary.csv",
    "Inventory_History": SAMPLE_DIR / "kapaye_inventory_history.csv",
}

BURGUNDY = "#711E32"
PALETTE = ["#711E32", "#B35B69", "#D7A4A9", "#62727B", "#D2A24C"]

st.set_page_config(
    page_title="KaPaye | Sales Intelligence",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f5f3; color: #292526; }
        [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e8e2df; }
        [data-testid="stMetric"] { background: #fff; border: 1px solid #eee8e5;
            padding: 16px 18px; border-radius: 12px; }
        [data-testid="stMetricLabel"] { color: #70686a; }
        h1, h2, h3 { color: #421722; }
        .eyebrow { color: #711e32; font-weight: 700; letter-spacing: .08em;
            text-transform: uppercase; font-size: .75rem; }
        .stAlert { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_data() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    try:
        if SAMPLE_WORKBOOK.exists():
            for name in DATASET_TYPES:
                tables[name] = load_dataset(SAMPLE_WORKBOOK, name)
        else:
            for name, path in SAMPLE_FILES.items():
                if path.exists():
                    tables[name] = load_dataset(path)
    except (OSError, ValueError, ImportError) as exc:
        st.error(f"Could not load the bundled synthetic sample dataset: {exc}")
    return tables


def ensure_state() -> None:
    if "tables" not in st.session_state:
        st.session_state.tables = initialize_data()
    if "table_sources" not in st.session_state:
        st.session_state.table_sources = {
            name: "Bundled synthetic sample data" for name in st.session_state.tables
        }


def money(value: float | int) -> str:
    return f"₱{value:,.2f}"


def format_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%b %d, %Y")


def chart_layout(fig: go.Figure, *, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=32, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#51494b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#ebe5e2", zeroline=False)
    return fig


def has_required_table(table_name: str) -> bool:
    return table_name in st.session_state.tables


def page_overview() -> None:
    st.markdown('<p class="eyebrow">Kapaye · emerging technology prototype</p>', unsafe_allow_html=True)
    st.title("Sales intelligence dashboard")
    st.caption(
        "A local, explainable sales forecasting prototype. Bundled records are synthetic "
        "sample data and do not represent actual KaPaye business results."
    )
    tables = st.session_state.tables
    daily = None
    try:
        if "Daily_Summary" in tables or "Sales_History" in tables:
            daily = prepare_daily_sales(tables)
    except ValueError as exc:
        st.warning(f"Forecast summary is unavailable: {exc}")
    if daily is None:
        st.info("Upload valid daily sales data or detailed sales records to populate this dashboard.")
        return
    try:
        products = (
            product_analytics(tables["Sales_History"], tables.get("Product_Master"))
            if "Sales_History" in tables else None
        )
        inventory = (
            inventory_analytics(tables["Inventory_History"], tables.get("Sales_History"))
            if "Inventory_History" in tables else None
        )
        insights = business_insights(daily, products, inventory)
    except ValueError as exc:
        st.warning(f"Some business insights are unavailable: {exc}")
        insights = []
    first, last = daily["date"].min(), daily["date"].max()
    total_revenue = float(daily["revenue"].sum())
    total_quantity = int(daily["quantity"].sum())
    best_day = daily.loc[daily["revenue"].idxmax()]
    a, b, c, d = st.columns(4)
    a.metric("Recorded revenue", money(total_revenue))
    b.metric("Units sold", f"{total_quantity:,}")
    c.metric("Daily average revenue", money(daily["revenue"].mean()))
    d.metric("History covered", f"{format_date(first)} – {format_date(last)}")
    left, right = st.columns([1.6, 1])
    with left:
        fig = px.area(
            daily, x="date", y="revenue", color_discrete_sequence=[BURGUNDY],
            labels={"date": "Date", "revenue": "Revenue (₱)"},
            title="Daily revenue history",
        )
        fig.update_traces(hovertemplate="%{x|%b %d, %Y}<br>Revenue: ₱%{y:,.2f}<extra></extra>")
        st.plotly_chart(chart_layout(fig), width="stretch")
    with right:
        weekday = daily.assign(day_type=daily["date"].dt.day_of_week.map(
            lambda day: "Weekend" if day >= 5 else "Weekday"
        )).groupby("day_type", as_index=False)["revenue"].mean()
        fig = px.bar(
            weekday, x="day_type", y="revenue", color="day_type",
            color_discrete_map={"Weekday": "#B35B69", "Weekend": BURGUNDY},
            labels={"day_type": "Day type", "revenue": "Average daily revenue (₱)"},
            title="Average sales by day type",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig), width="stretch")
        st.info(
            f"Highest recorded sales day: **{format_date(best_day['date'])}** "
            f"({money(best_day['revenue'])})."
        )
    if insights:
        st.subheader("Data-based business insights")
        for insight in insights:
            st.markdown(f"- {insight}")
    source_names = list(dict.fromkeys(st.session_state.table_sources.values()))
    st.caption("Data sources: " + "; ".join(source_names))


def page_upload() -> None:
    st.title("Upload historical data")
    st.write("Load a CSV or Excel workbook locally. Excel uploads can be read one worksheet at a time.")
    st.info(
        "Expected records are synthetic samples unless you upload your own data. "
        "Files are processed in memory and are not copied over the bundled dataset."
    )
    file = st.file_uploader("Choose an .xlsx or .csv file", type=["xlsx", "csv"])
    if file is None:
        st.caption("Required fields vary by sheet. Use the bundled workbook as a format example.")
        return
    sheet_name = None
    try:
        if file.name.lower().endswith(".xlsx"):
            sheets = pd.ExcelFile(file, engine="openpyxl").sheet_names
            if not sheets:
                st.error("This workbook has no worksheets.")
                return
            sheet_name = st.selectbox("Select worksheet", sheets)
        if st.button("Validate and load data", type="primary"):
            with st.spinner("Reading and validating the selected data..."):
                table = read_upload(file, sheet_name)
            dataset_type = table.attrs["dataset_type"]
            st.session_state.tables[dataset_type] = table
            st.session_state.table_sources[dataset_type] = f"Uploaded: {file.name}"
            st.success(
                f"Loaded {len(table):,} valid rows as "
                f"{table.attrs['dataset_type'].replace('_', ' ')}."
            )
            st.dataframe(table.head(10), width="stretch")
    except (ValueError, OSError, ImportError) as exc:
        st.error(f"Data could not be loaded. {exc}")


def page_forecasting() -> None:
    st.title("Sales forecasting")
    tables = st.session_state.tables
    if not ("Daily_Summary" in tables or "Sales_History" in tables):
        st.warning("Upload a Daily_Summary or Sales_History sheet before forecasting.")
        return
    try:
        daily = prepare_daily_sales(tables)
    except ValueError as exc:
        st.error(str(exc))
        return
    target_label = st.selectbox("Forecast target", ["Revenue (₱)", "Quantity sold (units)"])
    target = "revenue" if target_label.startswith("Revenue") else "quantity"
    horizon = st.select_slider("Forecast horizon", options=[7, 14, 30], value=7, format_func=lambda n: f"{n} days")
    if len(daily) < 30:
        st.warning(f"Only {len(daily)} daily observations are available. At least 30 are recommended.")
    try:
        with st.spinner("Training the forecasting model and generating future predictions..."):
            result = evaluate_model(daily, target)
            future = forecast_future(daily, result, target, horizon)
    except ValueError as exc:
        st.error(f"Forecast could not be generated. {exc}")
        return
    measure = "revenue" if target == "revenue" else "quantity"
    unit_label = "Revenue (₱)" if target == "revenue" else "Quantity (units)"
    history = daily.tail(60)
    history_line = go.Scatter(
        x=history["date"], y=history[target], name="Historical actual",
        mode="lines", line=dict(color="#8c8586", width=2),
        hovertemplate="%{x|%b %d, %Y}<br>Actual: %{y:,.2f}<extra></extra>",
    )
    forecast_line = go.Scatter(
        x=future["date"], y=future[measure], name="Forecast",
        mode="lines+markers", line=dict(color=BURGUNDY, width=3, dash="dash"),
        hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:,.2f}<extra></extra>",
    )
    fig = go.Figure([history_line, forecast_line])
    fig.update_layout(title=f"Next {horizon} days · {unit_label}", yaxis_title=unit_label, xaxis_title="Date")
    st.plotly_chart(chart_layout(fig, height=460), width="stretch")
    st.caption(
        "The dashed line is a model estimate, not a guarantee. Future lag features are "
        "updated with each prediction; actual future values are never used."
    )
    st.subheader("Forecast values")
    display = future[["date", measure]].rename(columns={measure: unit_label})
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display.style.format({unit_label: "₱{:,.2f}" if target == "revenue" else "{:,.2f}"}), width="stretch")
    if target == "quantity" and "revenue" in daily.columns:
        st.caption("Quantity is forecast separately from revenue; this table does not estimate a future peso amount.")


def page_evaluation() -> None:
    st.title("Model training and evaluation")
    tables = st.session_state.tables
    if not ("Daily_Summary" in tables or "Sales_History" in tables):
        st.warning("Upload daily or detailed sales data before evaluating the model.")
        return
    try:
        daily = prepare_daily_sales(tables)
        target_label = st.selectbox("Model target", ["Revenue (₱)", "Quantity sold (units)"])
        target = "revenue" if target_label.startswith("Revenue") else "quantity"
        with st.spinner("Building time features, fitting on the earlier observations, and evaluating the holdout..."):
            result = evaluate_model(daily, target)
    except ValueError as exc:
        st.error(str(exc))
        return
    metrics = result["metrics"]
    a, b, c = st.columns(3)
    suffix = "₱" if target == "revenue" else " units"
    a.metric("MAE", f"₱{metrics['mae']:,.2f}" if target == "revenue" else f"{metrics['mae']:,.2f} units")
    b.metric("RMSE", f"₱{metrics['rmse']:,.2f}" if target == "revenue" else f"{metrics['rmse']:,.2f} units")
    c.metric("R²", f"{metrics['r2']:.3f}" if metrics["r2"] is not None else "Not available")
    st.caption(f"Training observations: {result['train_size']:,} · Held-out observations: {result['test_size']:,} · Chronological split: earliest 80% / latest 20%.")
    st.markdown(
        """
        **How to explain these measures:** MAE is the average size of an error in the target's
        units. RMSE uses the same units but penalizes larger misses more. R² compares the
        predictions with a simple average-based reference; 1 is a perfect fit, 0 is roughly
        that reference, and negative values are worse. These test-set measures are evidence
        about this holdout only, not a promise of future accuracy.
        """
    )
    test = result["test_results"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["date"], y=test["actual"], name="Actual", mode="lines", line=dict(color="#777174")))
    fig.add_trace(go.Scatter(x=test["date"], y=test["predicted"], name="Predicted", mode="lines", line=dict(color=BURGUNDY)))
    fig.update_layout(title="Actual vs predicted · held-out period", xaxis_title="Date", yaxis_title=target_label)
    st.plotly_chart(chart_layout(fig, height=420), width="stretch")
    st.caption(
        "Validation is rolling one-day-ahead: each held-out date uses only earlier observed "
        "sales for its lag features. The model is fit only on the earlier training period."
    )
    st.dataframe(test, width="stretch")


def page_products() -> None:
    st.title("Product performance")
    if not has_required_table("Sales_History"):
        st.warning("Load the Sales_History worksheet to view product analytics.")
        return
    try:
        sales_source = st.session_state.table_sources.get("Sales_History")
        master_source = st.session_state.table_sources.get("Product_Master")
        matching_master = (
            st.session_state.tables.get("Product_Master")
            if sales_source is not None and sales_source == master_source
            else None
        )
        products = product_analytics(st.session_state.tables["Sales_History"], matching_master)
    except ValueError as exc:
        st.error(str(exc))
        return
    if products.empty:
        st.info("No valid product sales are available.")
        return
    top = products.iloc[0]
    a, b, c = st.columns(3)
    a.metric("Products recorded", f"{len(products):,}")
    b.metric("Top revenue product", str(top["product_name"]))
    c.metric("Top product revenue", money(float(top["revenue"])))
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            products.head(12).sort_values("quantity"), x="quantity", y="product_name",
            orientation="h", color_discrete_sequence=[BURGUNDY],
            labels={"quantity": "Units sold", "product_name": "Product"},
            title="Best-selling products by quantity",
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
    with right:
        fig = px.bar(
            products.head(12).sort_values("revenue"), x="revenue", y="product_name",
            orientation="h", color_discrete_sequence=["#B35B69"],
            labels={"revenue": "Revenue (₱)", "product_name": "Product"},
            title="Revenue by product",
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
    category = products.groupby("category", as_index=False)[["quantity", "revenue", "gross_profit"]].sum()
    fig = px.bar(
        category, x="category", y="revenue", color="category", color_discrete_sequence=PALETTE,
        labels={"category": "Category", "revenue": "Revenue (₱)"}, title="Sales performance by category",
    )
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.subheader("Product summary")
    st.dataframe(
        products.rename(columns={
            "product_name": "Product", "category": "Category", "quantity": "Units sold",
            "revenue": "Revenue (₱)", "gross_profit": "Gross profit (₱)",
            "unit_price": "Unit price (₱)", "unit_cost": "Unit cost (₱)",
        }).style.format({"Revenue (₱)": "₱{:,.2f}", "Gross profit (₱)": "₱{:,.2f}", "Unit price (₱)": "₱{:,.2f}", "Unit cost (₱)": "₱{:,.2f}"}),
        width="stretch",
    )


def page_inventory() -> None:
    st.title("Inventory analytics")
    if not has_required_table("Inventory_History"):
        st.warning("Load the Inventory_History worksheet to view stock analytics.")
        return
    demand = None
    if has_required_table("Sales_History"):
        demand = st.session_state.tables["Sales_History"]
    try:
        inventory = inventory_analytics(st.session_state.tables["Inventory_History"], demand)
    except ValueError as exc:
        st.error(str(exc))
        return
    if inventory.empty:
        st.info("No inventory records are available.")
        return
    low = int(inventory["low_stock"].sum())
    stockouts = int(inventory["stockout_days"].sum())
    a, b, c = st.columns(3)
    a.metric("Products tracked", f"{len(inventory):,}")
    a.caption("Opening and closing stock reflect each product's latest dated record.")
    b.metric("At/below reorder level", f"{low:,}")
    c.metric("Historical stockout records", f"{stockouts:,}")
    st.caption("Restock quantities are estimates based on observed average daily demand and a 7-day coverage target—not purchasing decisions.")
    fig = px.bar(
        inventory.sort_values("closing_stock"), x="closing_stock", y="product_name",
        orientation="h", color="low_stock",
        color_discrete_map={True: "#711E32", False: "#B9B1B0"},
        labels={"closing_stock": "Latest closing stock (units)", "product_name": "Product", "low_stock": "At/below reorder"},
        title="Current known stock by product",
    )
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.subheader("Suggested restocking priorities")
    st.dataframe(
        inventory.sort_values(["suggested_restock", "avg_daily_demand"], ascending=False).rename(columns={
            "product_name": "Product", "category": "Category", "opening_stock": "Latest opening stock",
            "closing_stock": "Latest closing stock", "reorder_level": "Reorder level",
            "avg_daily_demand": "Average daily demand", "stockout_days": "Historical stockout records",
            "suggested_restock": "Estimated restock (7-day coverage)", "low_stock": "At/below reorder",
        }).style.format({"Average daily demand": "{:,.2f}", "Estimated restock (7-day coverage)": "{:,.0f}"}),
        width="stretch",
    )


def page_explorer() -> None:
    st.title("Dataset explorer")
    tables = st.session_state.tables
    if not tables:
        st.info("No datasets loaded. Upload a CSV or Excel worksheet to begin.")
        return
    choice = st.selectbox("Dataset", list(tables))
    data = tables[choice]
    source = st.session_state.table_sources.get(choice, "Source not recorded")
    st.caption(f"{len(data):,} rows · {len(data.columns)} columns · {source}")
    search = st.text_input("Filter rows (text search)", placeholder="Type a value to filter across columns")
    view = data
    if search.strip():
        mask = data.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
        view = data.loc[mask]
    st.dataframe(view, width="stretch", height=520)
    st.download_button(
        "Download this table as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"{choice.lower()}.csv",
        mime="text/csv",
    )
    st.caption("Values marked `synthetic_simulated` are generated sample records, not actual business data.")


def main() -> None:
    apply_theme()
    ensure_state()
    with st.sidebar:
        st.markdown("## ☕ KaPaye")
        st.caption("Machine learning sales demo")
        page = st.radio(
            "Navigation",
            [
                "Dashboard Overview",
                "Upload Historical Data",
                "Sales Forecasting",
                "Model Training and Evaluation",
                "Product Performance",
                "Inventory Analytics",
                "Dataset Explorer",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Local prototype · synthetic data")
    routes = {
        "Dashboard Overview": page_overview,
        "Upload Historical Data": page_upload,
        "Sales Forecasting": page_forecasting,
        "Model Training and Evaluation": page_evaluation,
        "Product Performance": page_products,
        "Inventory Analytics": page_inventory,
        "Dataset Explorer": page_explorer,
    }
    routes[page]()


if __name__ == "__main__":
    main()
