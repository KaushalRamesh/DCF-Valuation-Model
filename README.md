# DCF Analyzer — Universal Stock Valuation Dashboard

**DCF Analyzer** is a local Python & Streamlit financial analysis and intrinsic valuation dashboard supporting **Indian NSE (`.NS`)** and **US equities** using live financial statements via `yfinance`.

> **Disclaimer:** This is an analytical and educational financial modeling tool, not investment advice or a guaranteed stock-price prediction system. Intrinsic value estimates depend directly on user assumptions and reported financial statements.

---

## Features

- **Universal Stock Search:** Supports Indian NSE stocks (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`) and US stocks (`AAPL`, `MSFT`, `GOOGL`) with graceful error handling for invalid or incomplete tickers.
- **Automated Financial Retrieval (3–5 Years):** Retrieves Income Statement, Cash Flow Statement, Balance Sheet, and Market Data via `yfinance` without paid APIs.
- **Strict Missing-Data Transparency:** Never invents or silently zeroes out missing metrics. Displays `Unavailable`, flags data provenance (`Reported`, `Calculated`, `Manual`, `Unavailable`), and lets users supply manual overrides.
- **Data Validation Layer:** Flags missing values, zero/negative share counts, currency mismatches (`currency` vs `financialCurrency`), negative FCF/EBIT, and extreme historical swings.
- **Historical Financial Analysis:** Computes YoY growth rates, Gross/EBIT/Net/FCF margins, CapEx % of Revenue, D&A % of Revenue, Effective Tax Rate, Operating NWC, and FCF trend.
- **5-Year Editable Forecast & Scenarios:** Generates non-naive initial assumptions (blended historical CAGR/medians with mean-reversion fade) across **Conservative**, **Base**, and **Optimistic** scenarios, with full per-year editing.
- **CAPM & WACC Engine:** Market-value-weighted WACC (`Market Cap`, never book equity) with selectable **India (INR)** and **United States (USD)** risk-free rate and equity risk premium presets.
- **Gordon Growth Terminal Value & Margin of Safety:** Validates `Discount Rate > Terminal Growth`, displays PV of Terminal Value separately, warns when TV > 75% of total value, and computes a Margin-of-Safety suggested purchase price.
- **Industry-Specific Valuation:** Automatically detects Banks, Insurers, and Financial Institutions, warns that standard FCFF Enterprise Value DCF may be inappropriate, and switches to **FCFE (Free Cash Flow to Equity)** discounted at Cost of Equity.
- **Interactive Plotly Charts & Sensitivity Heatmap:** Dark-theme analyst dashboard with currency-aware scaling (`₹ Crore` for INR, `$ Millions` for USD) and a live WACC vs Terminal Growth sensitivity heatmap.

---

## Project Structure

```text
dcf-analyzer/
├── app.py                     # Main Streamlit dashboard
├── requirements.txt           # Python dependencies
├── run.bat                    # Windows one-click launcher
├── .env.example               # Optional environment configuration template
├── .streamlit/
│   └── config.toml            # Dark financial dashboard theme
├── src/
│   ├── config.py              # Env loading and currency formatting helpers
│   ├── data_fetcher.py        # yfinance retrieval & manual override provenance
│   ├── financial_analysis.py  # Historical growth, margins, NWC, and FCF trends
│   ├── forecasting.py         # 5-year FCFF & FCFE projections and scenarios
│   ├── wacc.py                # CAPM Cost of Equity & market-weighted WACC
│   ├── dcf_engine.py          # Discounting, Gordon Growth TV, EV/Equity bridge
│   ├── sensitivity.py         # 2D WACC vs Terminal Growth matrix
│   ├── validation.py          # Data & assumption validation checks
│   └── industry.py            # Financial vs non-financial methodology detection
├── components/
│   ├── sidebar.py             # Sidebar controls & manual data overrides
│   ├── metrics.py             # Header, KPI cards, WACC & validation banners
│   ├── charts.py              # 7 interactive Plotly visualizations
│   └── tables.py              # Formatted financial, forecast & sensitivity tables
└── tests/
    ├── test_dcf_engine.py     # FCFF, FCFE, TV, EV, Equity & MoS unit tests
    ├── test_wacc.py           # CAPM & WACC unit tests
    ├── test_sensitivity.py    # Sensitivity matrix & edge-case tests
    └── test_edge_cases.py     # Missing data, currency mismatch & ticker tests
```

---

## Installation & Running on Windows (VS Code)

### Prerequisites
- **Python 3.11+** installed and on `PATH`.

### Setup Steps
1. Open the `dcf-analyzer` folder in **VS Code**.
2. Open a PowerShell terminal in VS Code and create/activate a virtual environment (optional but recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. *(Optional)* Copy `.env.example` to `.env` if you want to customize default risk-free rates, ERPs, or cache TTL.

### Running the Application
- **Using Streamlit CLI:**
  ```powershell
  streamlit run app.py
  ```
- **Or double-click / run `run.bat`:**
  ```cmd
  run.bat
  ```

### Running Unit Tests
```powershell
pytest -v
```

---

## How to Enter Stock Tickers

- **US Stocks:** Enter the standard ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`, `JPM`).
- **Indian NSE Stocks:** Append `.NS` (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`).
- **Indian BSE Stocks:** Append `.BO` (e.g., `RELIANCE.BO`).

---

## DCF Formulas Used

1. **Unlevered Free Cash Flow (UFCF):**
   $$\text{UFCF} = \text{EBIT} \times (1 - \text{Tax Rate}) + \text{D\&A} - \text{CapEx} - \Delta \text{NWC}$$
2. **CAPM & WACC:**
   $$K_e = R_f + \beta \times \text{ERP}$$
   $$\text{WACC} = \frac{E}{D + E} K_e + \frac{D}{D + E} K_d (1 - \text{Tax Rate})$$
   where $E$ is **Market Capitalization** (not book equity) and $D$ is Total Debt.
3. **Gordon Growth Terminal Value:**
   $$\text{TV} = \frac{\text{FCF}_N \times (1 + g)}{r - g} \quad (\text{requires } r > g)$$
4. **Enterprise Value to Equity Value (FCFF Mode):**
   $$\text{EV} = \sum_{t=1}^{5} \frac{\text{UFCF}_t}{(1 + \text{WACC})^t} + \frac{\text{TV}}{(1 + \text{WACC})^5}$$
   $$\text{Equity Value} = \text{EV} + \text{Cash} - \text{Total Debt} + \text{Non-Operating Adjustments}$$
   $$\text{Intrinsic Value / Share} = \frac{\text{Equity Value}}{\text{Shares Outstanding}}$$
5. **Financial Institutions (FCFE Mode):**
   Discounts projected Free Cash Flow to Equity directly at Cost of Equity ($K_e$) to estimate Equity Value without forcing an Enterprise Value bridge.

---

## Data Source Limitations

- Financial statements and market quotes are retrieved from Yahoo Finance (`yfinance`), which can occasionally have missing line items, reporting delays, or differing line-item classifications across jurisdictions.
- For ADRs or dual-listed companies where trading `currency` differs from `financialCurrency`, the validation layer raises an error to prevent silent currency mixing.
- Always verify critical inputs (Beta, Total Debt, Cash, Shares Outstanding, and CapEx) against official annual filings (`10-K` / Annual Report) before making any financial decisions.
