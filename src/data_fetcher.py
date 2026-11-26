"""Financial data retrieval layer using yfinance with strict missing-data transparency."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf


class TickerDataError(Exception):
    """Raised when a ticker is invalid or no usable market/financial data exists."""


@dataclass
class CompanyFinancialData:
    """Container for retrieved market data, historical statements, and provenance."""

    ticker: str
    market_data: dict[str, Any]
    income_statement: pd.DataFrame
    cashflow_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    provenance: dict[str, str] = field(default_factory=dict)
    raw_currency_mismatch: bool = False


INCOME_LINE_ITEMS: list[tuple[str, list[str]]] = [
    ("Revenue", ["Total Revenue", "Operating Revenue"]),
    ("Cost of Revenue", ["Cost Of Revenue", "Reconciled Cost Of Revenue"]),
    ("Gross Profit", ["Gross Profit"]),
    ("Operating Income / EBIT", ["Operating Income", "EBIT", "Total Operating Income As Reported"]),
    (
        "Net Income",
        [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income From Continuing Operation Net Minority Interest",
        ],
    ),
    ("Income Tax Expense", ["Tax Provision", "Income Tax Expense"]),
    ("Pretax Income", ["Pretax Income"]),
    ("Interest Expense", ["Interest Expense", "Interest Expense Non Operating"]),
]

CASHFLOW_LINE_ITEMS: list[tuple[str, list[str]]] = [
    (
        "Depreciation & Amortization",
        ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation"],
    ),
    ("Capital Expenditure", ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"]),
    ("Operating Cash Flow", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
    ("Free Cash Flow", ["Free Cash Flow"]),
    ("Change in Working Capital", ["Change In Working Capital"]),
]

BALANCE_LINE_ITEMS: list[tuple[str, list[str]]] = [
    (
        "Total Cash & Equivalents",
        ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"],
    ),
    ("Total Debt", ["Total Debt"]),
    ("Current Debt", ["Current Debt And Capital Lease Obligation", "Current Debt"]),
    ("Current Assets", ["Current Assets", "Total Current Assets"]),
    ("Current Liabilities", ["Current Liabilities", "Total Current Liabilities"]),
    (
        "Total Equity",
        ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    ),
]


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _normalize_columns(df: pd.DataFrame, max_years: int = 5) -> pd.DataFrame:
    """Sort statement columns chronologically (oldest -> newest) and format as FY YYYY."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = list(df.columns)
    try:
        col_dates = pd.to_datetime(cols)
        order = np.argsort(col_dates)
        df = df.iloc[:, order]
        if len(df.columns) > max_years:
            df = df.iloc[:, -max_years:]
        new_cols = [f"FY{pd.to_datetime(c).year}" for c in df.columns]
        # Handle duplicate fiscal years if restated/split
        seen: dict[str, int] = {}
        unique_cols: list[str] = []
        for label in new_cols:
            count = seen.get(label, 0) + 1
            seen[label] = count
            unique_cols.append(label if count == 1 else f"{label} ({count})")
        df.columns = unique_cols
    except Exception:
        if len(df.columns) > max_years:
            df = df.iloc[:, -max_years:]
        df.columns = [str(c) for c in df.columns]
    return df


def _extract_standardized_table(
    raw_df: pd.DataFrame,
    spec: list[tuple[str, list[str]]],
    provenance: dict[str, str],
    prefix: str,
    normalize_capex_positive: bool = False,
) -> pd.DataFrame:
    """Extract standardized line items from a yfinance DataFrame without fabricating missing values."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(index=[item[0] for item in spec], dtype=float)

    df = _normalize_columns(raw_df, max_years=5)
    # Build case-insensitive lookup of row names
    row_lookup = {str(idx).strip().lower(): idx for idx in df.index}
    result = pd.DataFrame(index=[item[0] for item in spec], columns=df.columns, dtype=float)

    for std_name, candidates in spec:
        matched_idx = None
        for cand in candidates:
            key = cand.strip().lower()
            if key in row_lookup:
                matched_idx = row_lookup[key]
                break
        if matched_idx is not None:
            series = pd.to_numeric(df.loc[matched_idx], errors="coerce")
            if std_name == "Capital Expenditure" and normalize_capex_positive:
                # yfinance reports CapEx as negative cash outflow; normalize to positive outflow magnitude
                series = series.abs()
            result.loc[std_name] = series
            provenance[f"{prefix}:{std_name}"] = "Reported" if series.notna().any() else "Unavailable"
        else:
            result.loc[std_name] = np.nan
            provenance[f"{prefix}:{std_name}"] = "Unavailable"

    return result


def fetch_company_financials(ticker: str) -> CompanyFinancialData:
    """Retrieve market data and 3–5 years of historical statements for a stock ticker via yfinance.

    Raises TickerDataError with a clear explanation if the ticker is invalid or no data is available.
    """
    clean_ticker = (ticker or "").strip().upper()
    if not clean_ticker:
        raise TickerDataError("Please enter a valid stock ticker (e.g., RELIANCE.NS, TCS.NS, AAPL, MSFT).")

    try:
        stock = yf.Ticker(clean_ticker)
        info = stock.info or {}
    except Exception as exc:
        raise TickerDataError(f"Failed to retrieve market data for '{clean_ticker}': {exc}") from exc

    # Verify ticker actually exists by checking price/history or company metadata
    current_price = (
        _safe_float(info.get("currentPrice"))
        or _safe_float(info.get("regularMarketPrice"))
        or _safe_float(info.get("previousClose"))
    )

    if current_price is None:
        try:
            hist = stock.history(period="5d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                current_price = _safe_float(hist["Close"].dropna().iloc[-1])
        except Exception:
            pass

    company_name = info.get("longName") or info.get("shortName")

    try:
        raw_inc = stock.financials
        raw_cf = stock.cashflow
        raw_bs = stock.balance_sheet
    except Exception as exc:
        raise TickerDataError(f"Failed to retrieve financial statements for '{clean_ticker}': {exc}") from exc

    has_any_statements = any(
        df is not None and not df.empty for df in (raw_inc, raw_cf, raw_bs)
    )
    if current_price is None and not company_name and not has_any_statements:
        raise TickerDataError(
            f"No market or financial data found for ticker '{clean_ticker}'. "
            "Verify the symbol (use .NS for NSE India stocks, e.g., RELIANCE.NS, or standard symbols for US stocks like AAPL)."
        )

    provenance: dict[str, str] = {}

    shares_outstanding = _safe_float(info.get("sharesOutstanding")) or _safe_float(
        info.get("impliedSharesOutstanding")
    )
    market_cap = _safe_float(info.get("marketCap"))
    if market_cap is not None:
        provenance["market:market_cap"] = "Reported"
    elif current_price is not None and shares_outstanding is not None:
        market_cap = current_price * shares_outstanding
        provenance["market:market_cap"] = "Calculated"
    else:
        provenance["market:market_cap"] = "Unavailable"

    beta = _safe_float(info.get("beta"))
    provenance["market:current_price"] = "Reported" if current_price is not None else "Unavailable"
    provenance["market:shares_outstanding"] = "Reported" if shares_outstanding is not None else "Unavailable"
    provenance["market:beta"] = "Reported" if beta is not None else "Unavailable"

    trade_currency = info.get("currency")
    financial_currency = info.get("financialCurrency") or trade_currency

    market_data: dict[str, Any] = {
        "ticker": clean_ticker,
        "company_name": company_name or clean_ticker,
        "exchange": info.get("fullExchangeName") or info.get("exchange") or "Unavailable",
        "sector": info.get("sector") or "Unavailable",
        "industry": info.get("industry") or "Unavailable",
        "current_price": current_price,
        "market_cap": market_cap,
        "shares_outstanding": shares_outstanding,
        "beta": beta,
        "currency": trade_currency or "USD",
        "financial_currency": financial_currency or trade_currency or "USD",
        "last_updated": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source": "Yahoo Finance (yfinance)",
    }

    income_df = _extract_standardized_table(raw_inc, INCOME_LINE_ITEMS, provenance, "inc")
    cashflow_df = _extract_standardized_table(
        raw_cf, CASHFLOW_LINE_ITEMS, provenance, "cf", normalize_capex_positive=True
    )
    balance_df = _extract_standardized_table(raw_bs, BALANCE_LINE_ITEMS, provenance, "bs")

    # Fallback derivations ONLY from reported components (never fabricated) and explicitly marked Calculated
    if not income_df.empty and "Gross Profit" in income_df.index:
        if income_df.loc["Gross Profit"].isna().all():
            rev = income_df.loc["Revenue"]
            cor = income_df.loc["Cost of Revenue"]
            both_present = rev.notna() & cor.notna()
            if both_present.any():
                income_df.loc["Gross Profit"] = np.where(both_present, rev - cor, np.nan)
                provenance["inc:Gross Profit"] = "Calculated"

    if not cashflow_df.empty and "Free Cash Flow" in cashflow_df.index:
        if cashflow_df.loc["Free Cash Flow"].isna().all():
            ocf = cashflow_df.loc["Operating Cash Flow"]
            capex = cashflow_df.loc["Capital Expenditure"]
            both_present = ocf.notna() & capex.notna()
            if both_present.any():
                cashflow_df.loc["Free Cash Flow"] = np.where(both_present, ocf - capex, np.nan)
                provenance["cf:Free Cash Flow"] = "Calculated"

    if not balance_df.empty and "Total Debt" in balance_df.index and raw_bs is not None and not raw_bs.empty:
        if balance_df.loc["Total Debt"].isna().all():
            norm_bs = _normalize_columns(raw_bs, max_years=5)
            row_lookup = {str(idx).strip().lower(): idx for idx in norm_bs.index}
            lt_key = row_lookup.get("long term debt")
            st_key = row_lookup.get("current debt") or row_lookup.get("current debt and capital lease obligation")
            if lt_key is not None or st_key is not None:
                lt_vals = pd.to_numeric(norm_bs.loc[lt_key], errors="coerce") if lt_key else pd.Series(0.0, index=norm_bs.columns)
                st_vals = pd.to_numeric(norm_bs.loc[st_key], errors="coerce") if st_key else pd.Series(0.0, index=norm_bs.columns)
                has_data = (lt_vals.notna() if lt_key else False) | (st_vals.notna() if st_key else False)
                if isinstance(has_data, pd.Series) and has_data.any():
                    balance_df.loc["Total Debt"] = lt_vals.fillna(0.0) + st_vals.fillna(0.0)
                    provenance["bs:Total Debt"] = "Calculated"

    currency_mismatch = bool(
        trade_currency
        and financial_currency
        and trade_currency.strip().upper() != financial_currency.strip().upper()
    )

    return CompanyFinancialData(
        ticker=clean_ticker,
        market_data=market_data,
        income_statement=income_df,
        cashflow_statement=cashflow_df,
        balance_sheet=balance_df,
        provenance=provenance,
        raw_currency_mismatch=currency_mismatch,
    )


def apply_manual_overrides(
    data: CompanyFinancialData,
    market_overrides: dict[str, Optional[float]],
    statement_overrides: dict[str, Optional[float]],
) -> CompanyFinancialData:
    """Return a copy of CompanyFinancialData with user-entered manual overrides applied and flagged.

    statement_overrides keys use the format '<prefix>:<Line Item>' where prefix is 'inc', 'cf', or 'bs',
    applying the manual override to the most recent fiscal period column.
    """
    new_market = dict(data.market_data)
    new_prov = dict(data.provenance)
    new_inc = data.income_statement.copy()
    new_cf = data.cashflow_statement.copy()
    new_bs = data.balance_sheet.copy()

    for key, val in market_overrides.items():
        if val is not None and not np.isnan(val):
            new_market[key] = float(val)
            new_prov[f"market:{key}"] = "Manual"

    # Recompute market cap if price or shares were manually updated and market_cap was not directly overridden
    if (
        ("current_price" in market_overrides or "shares_outstanding" in market_overrides)
        and market_overrides.get("market_cap") is None
        and new_market.get("current_price") is not None
        and new_market.get("shares_outstanding") is not None
    ):
        new_market["market_cap"] = float(new_market["current_price"]) * float(new_market["shares_outstanding"])
        new_prov["market:market_cap"] = "Calculated (from Manual)"

    table_map = {"inc": new_inc, "cf": new_cf, "bs": new_bs}
    for full_key, val in statement_overrides.items():
        if val is None or np.isnan(val) or ":" not in full_key:
            continue
        prefix, line_item = full_key.split(":", 1)
        df = table_map.get(prefix)
        if df is not None and not df.empty and line_item in df.index:
            latest_col = df.columns[-1]
            df.loc[line_item, latest_col] = float(val)
            new_prov[full_key] = "Manual"

    return CompanyFinancialData(
        ticker=data.ticker,
        market_data=new_market,
        income_statement=new_inc,
        cashflow_statement=new_cf,
        balance_sheet=new_bs,
        provenance=new_prov,
        raw_currency_mismatch=data.raw_currency_mismatch,
    )
