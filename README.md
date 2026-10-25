# KaPaye Machine Learning Sales Demo

A standalone Streamlit prototype for demonstrating local sales forecasting and business analytics. It is intentionally separate from the KaPaye application. The workbook and CSVs bundled in `sales_data.csv/` are **synthetic sample records**; they are not actual business data.

## Requirements

- Windows 10/11
- Python 3.11 or 3.12 recommended
- Internet access only to install Python packages (the app itself makes no external AI/API calls)

## Install and run (PowerShell)

Run these commands from this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks virtual environment activation for the current window, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then open the local URL shown by Streamlit (typically `http://localhost:8501`). In VS Code, open this folder and select `.venv` as the Python interpreter.

## Dashboard sections

1. **Dashboard Overview** — daily totals, historical revenue, and weekday/weekend comparison.
2. **Upload Historical Data** — upload a `.xlsx` workbook or `.csv`; select an Excel worksheet, validate its schema, and load it without overwriting bundled files.
3. **Sales Forecasting** — select daily revenue or units, choose 7, 14, or 30 days, and view recursive predictions alongside recent history.
4. **Model Training and Evaluation** — inspect a chronological 80/20 holdout, MAE, RMSE, R², and actual versus predicted values.
5. **Product Performance** — best-selling products, revenue, gross profit, and category sales.
6. **Inventory Analytics** — latest recorded stock, historical stockout counts, low-stock flags, and estimated 7-day restocking quantities.
7. **Dataset Explorer** — browse/filter each loaded table and download it as CSV.

The bundled workbook is loaded automatically when present. You can also upload only a particular worksheet as a CSV. Dataset detection supports the provided columns and common alternatives. Sales forecasting requires a `Daily_Summary` or `Sales_History` dataset; product analysis uses detailed sales records; inventory analysis uses inventory history. Invalid required dates or numeric fields produce an explicit error rather than being silently interpreted.

## ML workflow and limitations

- Daily sales are aggregated by calendar date and sorted before feature creation.
- Features include weekday, month, day of month, weekend, previous-day sales, sales seven days earlier, and a trailing seven-observation mean shifted so it excludes the current target.
- A Random Forest Regressor (fixed random seed) is fitted on the earliest 80% of feature-ready rows; the latest 20% is held out. No random shuffle is used.
- Held-out scores are calculated only from predictions for that later period. In the holdout, each prediction uses sales observed before that date (rolling one-day-ahead evaluation).
- Future forecasts are generated recursively. Each prediction becomes a lag input for subsequent future dates; future actual sales are never read.
- MAE and RMSE are expressed in the target's units. R² is omitted if the test set has fewer than two records. These measures describe this dataset and holdout only; they do not establish guaranteed accuracy.
- Restocking estimates use average observed daily product demand and a 7-day target coverage. They are estimates, not confirmed purchase recommendations.

## Classroom demonstration

1. Start the app and point out that the bundled records are synthetic.
2. Open **Dataset Explorer** to inspect the four source tables and date coverage.
3. Show product and inventory charts, then identify the computed high-sales day and restock candidates.
4. Open **Model Training and Evaluation** and explain the chronological holdout and error measures.
5. Open **Sales Forecasting**, select a horizon, and explain how predictions are fed forward to create later lag values.
6. Optionally upload a copy of the workbook or one of the CSVs and demonstrate worksheet selection and validation.

## Troubleshooting

- **`py` or Python is not found:** install Python 3.11/3.12 from python.org and enable the Python Launcher; reopen PowerShell.
- **Activation is blocked:** use the process-scoped execution policy command above, or run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\streamlit.exe run app.py` without activation.
- **`ModuleNotFoundError`:** activate the intended `.venv`, then run `python -m pip install -r requirements.txt`.
- **Excel engine or workbook error:** ensure `openpyxl` is installed and the file is a readable `.xlsx` workbook (not a password-protected file).
- **Missing columns:** compare the upload with the example sheets in `sales_data.csv/KaPaye_ML_Historical_Sales_Dataset.xlsx`. Required columns and actual found columns are included in the error where possible.
- **Not enough observations:** provide at least 17 daily observations for lag features and a usable holdout; complete records over a longer period are recommended.
- **Port 8501 is busy:** run `streamlit run app.py --server.port 8502`.

## Project files

- `app.py` — Streamlit navigation, upload flow, dashboard, plots, and tables.
- `data_processor.py` — file reading, schema validation, normalization, and daily aggregation.
- `train_model.py` — leakage-aware feature creation, chronological evaluation, and recursive forecasting.
- `analytics.py` — product, inventory, and data-driven business insights.
- `data/` — optional location for local datasets; supplied files are retained in `sales_data.csv/`.
- `models/` — reserved for future model persistence; this prototype does not save dataset-specific model artifacts.
