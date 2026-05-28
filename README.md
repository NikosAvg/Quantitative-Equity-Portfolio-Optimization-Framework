# Quantitative Momentum + Low Volatility Portfolio Strategy

## Overview

This project implements a fully systematic equity portfolio construction and backtesting framework using:

- Momentum factor investing
- Low-volatility factor investing
- Sector diversification constraints
- Mean-variance portfolio optimization
- Walk-forward validation
- Transaction cost modeling
- Dynamic portfolio rebalancing

The system downloads historical market data and company fundamentals for the S&P 500 universe, ranks securities using factor signals, constructs optimized portfolios, and evaluates performance against the SPY benchmark.

---

# Strategy Philosophy

The strategy combines two well-known quantitative equity factors:

| Factor | Goal |
|---|---|
| Momentum | Buy stocks with strong medium-term price trends |
| Low Volatility | Prefer stocks with lower realized volatility |

The final portfolio is optimized to maximize risk-adjusted returns while controlling turnover and concentration risk.

---

# Main Features

## Universe Selection

- Uses the full S&P 500 stock universe
- Tickers are downloaded dynamically from Wikipedia
- Automatic fallback source if Wikipedia fails

---

## Factor Model

### Momentum Factor

Momentum is calculated using:

- 12-month lookback
- 1-month skip period

Formula:

```python
momentum = shifted_price / past_price - 1
```

This avoids short-term mean reversion effects.

---

### Low Volatility Factor

Low-volatility scores are calculated using annualized rolling volatility:

```python
volatility = returns.rolling(window).std() * sqrt(252)
```

Assets with lower volatility receive higher scores.

---

### Cross-Sectional Normalization

Each factor undergoes:

1. Winsorization
2. Z-score normalization

This improves robustness against outliers.

---

## Portfolio Construction

### Stock Selection

The strategy:

1. Computes combined factor scores
2. Ranks all stocks
3. Selects the top N names
4. Applies sector diversification constraints

Combined score:

```python
score = 0.6 * momentum + 0.4 * low_volatility
```

---

### Sector Constraints

To avoid over-concentration:

- No sector can exceed:

```python
MAX_PER_SECTOR_RATIO
```

---

# Optimization Engine

## Objective

The optimizer maximizes risk-adjusted returns using a modified Sharpe-ratio objective:

```python
objective = -sharpe + turnover_penalty
```

---

## Covariance Estimation

The optimizer uses:

```python
LedoitWolf()
```

for covariance shrinkage estimation.

Fallback:

```python
np.cov()
```

is used if shrinkage estimation fails.

---

## Optimization Constraints

### Long-only Portfolio

```python
0 <= weight <= MAX_POSITION_WEIGHT
```

No short selling allowed.

---

### Full Investment Constraint

Portfolio weights must sum to 100%:

```python
sum(weights) = 1
```

---

# Backtesting Framework

## Monthly Rebalancing

The strategy rebalances monthly using:

```python
prices.resample("M").last()
```

---

## Transaction Costs

The model includes:

- Commission costs
- Slippage costs

Configured using:

```python
COST_BPS
SLIPPAGE_BPS
```

---

# Walk-Forward Validation

The framework uses rolling out-of-sample testing:

| Parameter | Default |
|---|---|
| Training Window | 10 years |
| Test Window | 2 years |

This avoids overfitting and simulates realistic deployment.

---

# Benchmark Comparison

The strategy compares performance against:

```python
SPY
```

---

# Current Portfolio Integration

The framework supports real-world portfolio management through:

```python
CURRENT_SHARES
INITIAL_CASH
```

Existing holdings are incorporated into optimization.

---

# Risk Controls

## Position Limits

```python
MAX_POSITION_WEIGHT
```

---

## Sector Diversification

```python
MAX_PER_SECTOR_RATIO
```

---

## Trade Thresholding

Small trades below:

```python
MIN_TRADE_THRESHOLD
```

are ignored to reduce frictional costs.

---

# Data Sources

## Price Data

Downloaded using:

```python
yfinance
```

---

## Fundamental Data

Retrieved per ticker:

- PE ratio
- Return on equity (ROE)
- Sector classification

---

# Dependencies

Install required packages:

```bash
pip install pandas numpy matplotlib scipy scikit-learn yfinance tqdm
```

---

# Example Usage

Run the strategy:

```bash
python strategy.py
```

---

# Disclaimer

This project is for research and educational purposes only.

It is not financial advice.

Past performance does not guarantee future results.

Live trading involves substantial financial risk.
