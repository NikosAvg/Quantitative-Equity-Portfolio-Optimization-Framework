import datetime as dt
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

# -----------------------------
# Parameters
# -----------------------------
#
#

# -----------------------------
# Existing Portfolio State
# -----------------------------
CURRENT_SHARES = {
    "WDC": 0.1286,
    "WBD": 6.5,
    "NEM": 1,
    "CIEN": 0.1244,
    "FIX": 0.047,
    "MU": 0.1573,
    "STX": 0.0426,
}

INITIAL_CASH = 128

# Optimization Controls
TURNOVER_PENALTY = 0.15
MAX_POSITION_WEIGHT = 0.25
MAX_WEIGHT_CHANGE = 0.10
MIN_TRADE_THRESHOLD = 0.01
ALLOW_CASH = False


START = "2005-01-01"
END = "2026-5-26"
LOOKBACK_MOM = 252
SKIP_DAYS = 21
LOOKBACK_VOL = 60
TOP_N = 5
MAX_PER_SECTOR_RATIO = 0.5
TARGET_VOL = 0.10
COST_BPS = 0.0003
SLIPPAGE_BPS = 0.0005
FUND_LAG_DAYS = 45
TRAIN_YEARS = 10
TEST_YEARS = 2
BATCH_SIZE = 80
MAX_WORKERS = 12
DOWNLOAD_RETRIES = 3
RETRY_SLEEP = 1.5

CACHE_DIR = "./cache"
PRICE_CACHE = os.path.join(CACHE_DIR, "prices.csv")
FUND_CACHE = os.path.join(CACHE_DIR, "fundamentals.csv")

np.random.seed(42)

# -----------------------------
# Utilities
# -----------------------------

# -----------------------------
# Portfolio State Utilities
# -----------------------------


def build_current_weights(current_shares, latest_prices, cash=0.0):
    values = {}

    total_value = cash

    for ticker, shares in current_shares.items():
        if ticker in latest_prices.index:
            position_value = shares * latest_prices[ticker]
            values[ticker] = position_value
            total_value += position_value

    if total_value <= 0:
        return pd.Series(dtype=float)

    weights = pd.Series(values) / total_value

    if cash > 0:
        weights["CASH"] = cash / total_value

    return weights


def apply_trade_threshold(target_weights, current_weights):
    current_weights = current_weights.reindex(target_weights.index).fillna(0)

    delta = target_weights - current_weights

    small_trades = delta.abs() < MIN_TRADE_THRESHOLD

    adjusted = target_weights.copy()
    adjusted[small_trades] = current_weights[small_trades]

    if adjusted.sum() > 0:
        adjusted = adjusted / adjusted.sum()

    return adjusted


def build_dynamic_bounds(columns, current_weights):
    bounds = []

    for ticker in columns:
        current_weight = current_weights.get(ticker, 0.0)

        lower = max(0.0, current_weight - MAX_WEIGHT_CHANGE)

        upper = min(
            MAX_POSITION_WEIGHT,
            current_weight + MAX_WEIGHT_CHANGE,
        )

        bounds.append((lower, upper))

    return bounds


def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def normalize_ticker(ticker: str) -> str:
    return ticker.replace(".", "-").strip()


def chunked(iterable, size):
    it = list(iterable)
    for i in range(0, len(it), size):
        yield it[i : i + size]


# # -----------------------------
# # Tickers
# # -----------------------------
def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url, header=0)[0]
        tickers = df["Symbol"].tolist()
        print(f"Loaded {len(tickers)} tickers from Wikipedia.")
    except Exception as e:
        print("Wikipedia failed, using GitHub fallback:", e)
        url2 = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df2 = pd.read_csv(url2)
        tickers = df2["Symbol"].tolist()
        print(f"Loaded {len(tickers)} tickers from GitHub fallback.")
    tickers = [normalize_ticker(t) for t in tickers]
    # print(tickers)
    return tickers


# -----------------------------
# Prices
# -----------------------------


def normalize_ticker(ticker: str) -> str:
    return ticker.replace(".", "-").strip()


def download_prices(tickers, start, end):
    ensure_cache_dir()

    # Normalize + deduplicate
    tickers = list(dict.fromkeys([normalize_ticker(t) for t in tickers]))

    # -----------------------------
    # Load cache
    # -----------------------------
    if os.path.exists(PRICE_CACHE):
        df = pd.read_csv(PRICE_CACHE, index_col=0)

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")

        missing = list(set(tickers) - set(df.columns))

        if missing:
            print(f"Cache missing {len(missing)} tickers, downloading them...")

            new_data = yf.download(
                missing, start=start, end=end, progress=True, auto_adjust=False
            )

            if new_data.empty:
                print("⚠️ No data returned from Yahoo for missing tickers")
            else:
                if isinstance(new_data.columns, pd.MultiIndex):
                    close = new_data["Close"]
                    new_prices = close.copy()
                else:
                    # Single ticker fallback
                    new_prices = new_data["Close"].to_frame()
                    new_prices.columns = missing

                df = df.join(new_prices, how="outer")

        # Clean cache
        df = df.sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")

        # Keep only requested tickers (safe)
        df = df.reindex(columns=tickers)

        # Drop tickers with no data at all
        df = df.dropna(axis=1, how="all")

        print("Final cached df shape:", df.shape)

        df.to_csv(PRICE_CACHE)

        return df

    # -----------------------------
    # Fresh download
    # -----------------------------
    print(f"Downloading {len(tickers)} tickers...")

    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)

    if data.empty:
        raise ValueError("Yahoo returned empty dataset")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        prices = close.copy()
    else:
        prices = data["Close"].to_frame()
        prices.columns = tickers

    # Clean
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()]
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Drop empty columns
    prices = prices.dropna(axis=1, how="all")

    print("Downloaded Data shape:", prices.shape)

    prices.to_csv(PRICE_CACHE)

    return prices


# -----------------------------
# Fundamentals
# -----------------------------
def _download_fund_one(t):
    try:
        tk = yf.Ticker(t)
        info = tk.info or {}
        return t, {
            "PE": info.get("trailingPE", np.nan),
            "ROE": info.get("returnOnEquity", np.nan),
            "Sector": info.get("sector", "Unknown"),
        }
    except:
        return t, {"PE": np.nan, "ROE": np.nan, "Sector": "Unknown"}


def download_fundamentals(tickers):
    ensure_cache_dir()

    out = {}

    print(f"Downloading fundamentals for {len(tickers)} tickers...")

    for t in tqdm(tickers, desc="Fundamentals", unit="ticker"):
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}

            out[t] = {
                "PE": info.get("trailingPE", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "Sector": info.get("sector", "Unknown"),
            }

        except Exception:
            out[t] = {"PE": np.nan, "ROE": np.nan, "Sector": "Unknown"}

    df = pd.DataFrame(out).T

    if df.empty:
        raise ValueError("Fundamentals download failed — empty DataFrame")

    df = df.reindex(tickers)

    df.to_csv(FUND_CACHE)

    print("\nFundamentals shape:", df.shape)

    return df


def get_next_month_top_n(prices, fundamentals):
    sector_map = fundamentals["Sector"].to_dict()

    # Ensure sorted index
    prices = prices.sort_index()

    # Latest available date
    asof = prices.index[-1]

    # Compute factors on full history
    mom_full = compute_momentum(prices)
    lv_full = compute_lowvol(prices)

    # Get latest available cross-section (aligned safely)
    mom = mom_full.loc[:asof].iloc[-1]
    lv = lv_full.loc[:asof].iloc[-1]

    # Cross-sectional transform
    mom_t = transform_cross_section(mom)
    lv_t = transform_cross_section(lv)

    # Combined signal
    score = 0.6 * mom_t + 0.4 * lv_t

    # Remove NaNs
    score = score.dropna()

    # Sector-constrained selection
    selected = select_top_n_with_sector(score, sector_map)

    # Build output
    result = pd.DataFrame(
        {"Ticker": selected, "Score": score.loc[selected].values}
    ).sort_values(by="Score", ascending=False)

    return result


# -----------------------------
# Factors
# -----------------------------


def compute_momentum(prices):
    shifted = prices.shift(SKIP_DAYS)
    past = shifted.shift(LOOKBACK_MOM - SKIP_DAYS)
    return shifted / past - 1.0


def compute_lowvol(prices):
    returns = prices.pct_change()
    vol = returns.rolling(LOOKBACK_VOL).std() * np.sqrt(252)
    return 1.0 / vol


def winsorize_series(s):
    if s.dropna().empty:
        return s
    return s.clip(s.quantile(0.01), s.quantile(0.99))


def zscore_cs(s):
    sd = s.std()
    if sd == 0 or pd.isna(sd):
        return s - s.mean()
    return (s - s.mean()) / sd


def transform_cross_section(s):
    return zscore_cs(winsorize_series(s))


# -----------------------------
# Selection
# -----------------------------


def select_top_n_with_sector(scores, sector_map):
    sorted_t = scores.sort_values(ascending=False)
    selected = []
    sector_count = {}
    max_per_sector = max(1, int(math.floor(TOP_N * MAX_PER_SECTOR_RATIO)))

    for t in sorted_t.index:
        sec = sector_map.get(t, "Unknown")
        if sector_count.get(sec, 0) < max_per_sector:
            selected.append(t)
            sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(selected) >= TOP_N:
            break

    return selected


# -----------------------------
# Optimization
# -----------------------------


def optimize_weights(
    returns,
    current_weights=None,
    turnover_penalty=TURNOVER_PENALTY,
):
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")

    columns = returns.columns
    n = len(columns)

    if n == 0:
        return pd.Series(dtype=float)

    # -------------------------
    # Clean current weights
    # -------------------------
    if current_weights is None:
        current_weights = pd.Series(np.ones(n) / n, index=columns)

    current_weights = current_weights.reindex(columns).fillna(0.0)

    # -------------------------
    # Stable mean/cov estimates
    # -------------------------
    mu = returns.mean().values * 252

    try:
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_ * 252
    except Exception:
        cov = np.cov(returns.values.T) * 252

    cov = np.nan_to_num(cov, nan=1e-6, posinf=1e-6, neginf=1e-6)

    # -------------------------
    # Objective (convex + stable)
    # -------------------------
    def objective(w):
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        port_ret = np.dot(w, mu)
        port_var = np.dot(w.T, np.dot(cov, w))
        port_vol = np.sqrt(max(port_var, 1e-12))

        sharpe = port_ret / port_vol

        turnover = np.sum(np.abs(w - current_weights.values))

        return -sharpe + turnover_penalty * turnover

    # -------------------------
    # Constraints
    # -------------------------
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    bounds = [(0.0, MAX_POSITION_WEIGHT)] * n

    w0 = current_weights.values
    if not np.isfinite(w0).all() or w0.sum() == 0:
        w0 = np.ones(n) / n

    # -------------------------
    # Solve
    # -------------------------
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    # -------------------------
    # Fallback safety
    # -------------------------
    if result.success and np.isfinite(result.x).all():
        final_weights = pd.Series(result.x, index=columns)
    else:
        final_weights = current_weights.copy()

    final_weights = final_weights.fillna(0.0)

    # normalize (safety)
    if final_weights.sum() > 0:
        final_weights = final_weights / final_weights.sum()
    else:
        final_weights = pd.Series(np.ones(n) / n, index=columns)

    return final_weights


# -----------------------------
# Walk-forward Validation
# -----------------------------


def walk_forward_splits(prices, train_years=TRAIN_YEARS, test_years=TEST_YEARS):
    # Ensure datetime index
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()]
    prices = prices.sort_index()

    dates = prices.index

    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("prices index is not DatetimeIndex")

    start = dates.min()
    splits = []

    current_train_start = start

    while True:
        train_end = current_train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        if test_end > dates.max():
            break

        splits.append((train_end, test_end))
        current_train_start = current_train_start + pd.DateOffset(years=test_years)

    return splits


# -----------------------------
# Backtest for a period
# -----------------------------


def run_backtest(prices, fundamentals, start_date, end_date):
    sector_map = fundamentals["Sector"].to_dict()

    prices = prices.loc[start_date:end_date]

    rebal_dates = prices.resample("M").last().index

    mom_full = compute_momentum(prices)
    lv_full = compute_lowvol(prices)

    pnl_history = []
    prev_weights = pd.Series(0.0, index=prices.columns)

    for i in range(len(rebal_dates) - 1):
        rd = rebal_dates[i]
        next_rd = rebal_dates[i + 1]

        idx_rd = prices.index.get_indexer([rd], method="ffill")[0]
        idx_next = prices.index.get_indexer([next_rd], method="ffill")[0]

        if idx_rd == -1 or idx_next == -1:
            continue

        asof = prices.index[idx_rd]

        mom = mom_full.loc[asof]
        lv = lv_full.loc[asof]

        score = 0.6 * transform_cross_section(mom) + 0.4 * transform_cross_section(lv)

        selected = select_top_n_with_sector(score.dropna(), sector_map)

        if selected:
            rets = prices[selected].pct_change().dropna().tail(LOOKBACK_VOL)
            w = optimize_weights(rets)
        else:
            w = pd.Series(0.0, index=prices.columns)

        weights = pd.Series(0.0, index=prices.columns)
        weights.loc[selected] = w.reindex(selected).fillna(0.0).values

        p_t = prices.loc[asof]
        p_t1 = prices.iloc[idx_next]
        period_ret = (p_t1 / p_t - 1.0).fillna(0)

        gross = (weights * period_ret).sum()
        turnover = (weights - prev_weights).abs().sum()
        tc = turnover * (COST_BPS + SLIPPAGE_BPS)
        net = gross - tc

        pnl_history.append(net)
        prev_weights = weights.copy()

    return pd.Series(pnl_history)


def run_full_backtest(prices, fundamentals):
    sector_map = fundamentals["Sector"].to_dict()

    prices = prices.sort_index()

    rebal_dates = prices.resample("M").last().index

    mom_full = compute_momentum(prices)
    lv_full = compute_lowvol(prices)

    equity_curve = []
    prev_weights = pd.Series(0.0, index=prices.columns)

    portfolio_value = 1.0  # start normalized

    for i in range(len(rebal_dates) - 1):
        rd = rebal_dates[i]
        next_rd = rebal_dates[i + 1]

        idx_rd = prices.index.get_indexer([rd], method="ffill")[0]
        idx_next = prices.index.get_indexer([next_rd], method="ffill")[0]

        if idx_rd == -1 or idx_next == -1:
            continue

        asof = prices.index[idx_rd]

        mom = mom_full.loc[asof]
        lv = lv_full.loc[asof]

        score = 0.6 * transform_cross_section(mom) + 0.4 * transform_cross_section(lv)

        selected = select_top_n_with_sector(score.dropna(), sector_map)

        if selected:
            rets = prices[selected].pct_change().dropna().tail(LOOKBACK_VOL)
            w = optimize_weights(rets)
        else:
            w = pd.Series(0.0, index=prices.columns)

        weights = pd.Series(0.0, index=prices.columns)
        weights.loc[selected] = w.reindex(selected).fillna(0.0).values

        # returns over next period
        p0 = prices.loc[asof]
        p1 = prices.iloc[idx_next]

        period_ret = (p1 / p0 - 1.0).fillna(0)

        gross = (weights * period_ret).sum()

        turnover = (weights - prev_weights).abs().sum()
        tc = turnover * (COST_BPS + SLIPPAGE_BPS)

        net = gross - tc

        portfolio_value *= 1 + net

        equity_curve.append(
            {"date": asof, "portfolio_value": portfolio_value, "period_return": net}
        )

        prev_weights = weights.copy()

    return pd.DataFrame(equity_curve).set_index("date")


def get_benchmark_equity(prices):
    spy = yf.download(
        "SPY", start=prices.index.min(), end=prices.index.max(), progress=False
    )["Close"]

    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]

    spy = spy.dropna()

    spy = spy.resample("M").last().dropna()

    spy_ret = spy.pct_change().fillna(0)
    equity = (1 + spy_ret).cumprod()

    return equity.to_frame("SPY")


def plot_performance(equity_df, benchmark_df=None):

    plt.figure(figsize=(12, 6))

    plt.plot(equity_df.index, equity_df["portfolio_value"], label="Strategy")

    if benchmark_df is not None:
        plt.plot(benchmark_df.index, benchmark_df.iloc[:, 0], label="SPY")

    plt.title("Portfolio Equity Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------
# Full validation pipeline
# -----------------------------


def validate_model(prices, fundamentals):
    splits = walk_forward_splits(prices)

    all_results = []

    for i, (train_end, test_end) in enumerate(splits):
        print("\n------------------------------")
        print(f"Fold {i + 1}/{len(splits)}")
        print(f"Train end: {train_end.date()} | Test end: {test_end.date()}")
        print("------------------------------")

        start_time = time.time()

        train_start = prices.index.min()

        pnl = run_backtest(prices, fundamentals, train_start, test_end)

        elapsed = time.time() - start_time
        print(f"Fold {i + 1} completed in {elapsed:.2f} seconds")

        if len(pnl) == 0:
            continue

        cum = (1 + pnl).prod() - 1
        sharpe = pnl.mean() / pnl.std() * np.sqrt(12) if pnl.std() > 0 else np.nan

        all_results.append({"fold": i + 1, "return": cum, "sharpe": sharpe})

    df = pd.DataFrame(all_results)

    print("\nValidation summary:")
    print(df)
    print("\nAverage Sharpe:", df["sharpe"].mean())
    print("Average Return:", df["return"].mean())

    return df


# -----------------------------
# Next Month Portfolio
# -----------------------------


def get_next_month_portfolio(prices, fundamentals, current_shares=None, cash=0.0):
    sector_map = fundamentals["Sector"].to_dict()
    prices = prices.sort_index()

    asof = prices.index[-1]
    latest_prices = prices.loc[asof]

    current_weights = build_current_weights(
        current_shares or {},
        latest_prices,
        cash,
    )
    # -----------------------------
    # Factor Signals
    # -----------------------------

    mom_full = compute_momentum(prices)
    lv_full = compute_lowvol(prices)

    mom = mom_full.loc[:asof].iloc[-1]
    lv = lv_full.loc[:asof].iloc[-1]

    score = 0.6 * transform_cross_section(mom) + 0.4 * transform_cross_section(lv)

    score = score.dropna()

    selected = select_top_n_with_sector(
        score,
        sector_map,
    )

    if len(selected) == 0:
        return None

    # Include already owned positions
    existing_positions = [t for t in current_weights.index if t != "CASH"]

    optimization_universe = list(set(selected + existing_positions))

    # -----------------------------
    # Return Matrix
    # -----------------------------

    rets = prices[optimization_universe].pct_change().tail(LOOKBACK_VOL)
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    rets = rets.dropna(axis=0, how="any")
    if ALLOW_CASH:
        rets["CASH"] = 0.0001 / 252

    # -----------------------------
    # Optimize
    # -----------------------------

    weights = optimize_weights(
        rets,
        current_weights=current_weights,
    )

    weights = weights / weights.sum()

    # -----------------------------
    # Portfolio Trades
    # -----------------------------

    current_aligned = current_weights.reindex(weights.index).fillna(0)

    trades = weights - current_aligned

    trades = trades[trades.abs() > 0]

    return {
        "asof": asof,
        "selected": selected,
        "weights": weights,
        "current_weights": current_weights,
        "trades": trades,
    }


def evaluate_next_month(prices, fundamentals):
    portfolio = get_next_month_portfolio(prices, fundamentals)

    if portfolio is None:
        return None

    asof = portfolio["asof"]
    weights = portfolio["weights"]

    # Align returns for next month
    next_month_idx = prices.index[prices.index > asof]
    if len(next_month_idx) == 0:
        return None

    next_date = next_month_idx[0]

    # Asset returns over next period
    p0 = prices.loc[asof, weights.index]
    p1 = prices.loc[next_date, weights.index]

    asset_returns = (p1 / p0 - 1).fillna(0)

    # Portfolio return
    port_ret = (weights * asset_returns).sum()

    # Benchmark return (SPY)
    spy = yf.download("SPY", start=asof, end=next_date, progress=False)["Close"]
    if len(spy) < 2:
        return None

    spy_ret = spy.iloc[-1] / spy.iloc[0] - 1

    alpha = port_ret - spy_ret

    return {
        "asof": asof,
        "next_date": next_date,
        "portfolio_return": port_ret,
        "benchmark_return": spy_ret,
        "alpha": alpha,
        "weights": weights,
    }


def sanitize_universe(df):
    if not ALLOW_CASH and "CASH" in df.columns:
        df = df.drop(columns=["CASH"])
    return df


# -----------------------------
# Main
# -----------------------------


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    tickers = get_sp500_tickers()
    prices = download_prices(tickers, START, END)
    fundamentals = download_fundamentals(prices.columns.tolist())

    # -----------------------------
    # Run validation (walk-forward)
    # -----------------------------
    print("\n==============================")
    print("Running walk-forward validation")
    print("==============================\n")

    validate_model(prices, fundamentals)

    print("\n==============================")
    print("FULL HISTORICAL PERFORMANCE")
    print("==============================\n")

    equity = run_full_backtest(prices, fundamentals)

    bench = get_benchmark_equity(prices)

    print(equity.tail())

    plot_performance(equity, bench)

    # -----------------------------
    # Build next-month portfolio
    # -----------------------------

    print("\n==============================")
    print("Next Month Portfolio")
    print("==============================\n")

    portfolio = get_next_month_portfolio(
        prices,
        fundamentals,
        current_shares=CURRENT_SHARES,
        cash=INITIAL_CASH,
    )

    if portfolio is None:
        print("Not enough data to build portfolio.")
        return

    asof = portfolio["asof"]
    weights = portfolio["weights"]
    trades = portfolio["trades"]

    print("As of date:", asof)
    print(weights)
    print("\nTarget Portfolio Weights:")
    print((weights * 100).round(2).sort_values(ascending=False).astype(str) + "%")

    print("\nRecommended Trades:")

    for ticker, trade in trades.sort_values(ascending=False).items():
        direction = "BUY" if trade > 0 else "SELL"

        print(f"{direction:4s} {ticker:6s} {abs(trade) * 100:.2f}%")


if __name__ == "__main__":
    main()
