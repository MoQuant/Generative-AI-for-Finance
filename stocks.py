"""
================================================================================
  MODERN PORTFOLIO THEORY — CIO QUANTITATIVE DASHBOARD
  Author : Quantitative Strategies Group
  Version: 1.0
  Usage  : python mpt_cio_dashboard.py [--csv path/to/data.csv]
            If no CSV is supplied, synthetic 5-year price data is generated.
================================================================================
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── colour palette (dark, institutional) ─────────────────────────────────────
BG        = "#0D1117"
PANEL     = "#161B22"
BORDER    = "#30363D"
GOLD      = "#D4A843"
TEAL      = "#3DD6C9"
CRIMSON   = "#E05C5C"
LAVENDER  = "#9B8FD4"
WHITE     = "#E6EDF3"
GREY      = "#8B949E"
GREEN     = "#3FB950"
ORANGE    = "#F78166"

RISK_FREE = 0.05          # annual risk-free rate
TRADING_DAYS = 252
N_PORTFOLIOS = 8_000      # Monte Carlo portfolios for the efficient frontier


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def load_prices(csv_path: str | None) -> pd.DataFrame:
    """Load price data or fall back to synthetic generation."""
    if csv_path:
        prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        prices = prices.ffill().dropna(axis=1, thresh=int(len(prices) * 0.8))
        prices = prices.select_dtypes(include=[np.number])
        print(f"[DATA] Loaded {prices.shape[1]} tickers × {prices.shape[0]} rows from {csv_path}")
    else:
        print("[DATA] No CSV supplied — generating synthetic 5-year daily price data …")
        prices = _generate_synthetic_prices()
    return prices


def _generate_synthetic_prices(
    tickers=("AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "XOM", "JNJ", "TSLA", "NVDA"),
    years=5,
    seed=42,
) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    n     = TRADING_DAYS * years
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    mu_ann    = rng.uniform(0.06, 0.28, len(tickers))
    sigma_ann = rng.uniform(0.15, 0.55, len(tickers))
    mu_daily  = mu_ann / TRADING_DAYS
    sig_daily = sigma_ann / np.sqrt(TRADING_DAYS)

    corr_base = 0.35 + rng.uniform(-0.05, 0.05, (len(tickers), len(tickers)))
    np.fill_diagonal(corr_base, 1.0)
    corr_base = (corr_base + corr_base.T) / 2
    # ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(corr_base)
    eigvals = np.clip(eigvals, 1e-8, None)
    corr_base = eigvecs @ np.diag(eigvals) @ eigvecs.T

    L        = np.linalg.cholesky(corr_base)
    z        = rng.standard_normal((n, len(tickers)))
    corr_z   = z @ L.T
    log_ret  = mu_daily + sig_daily * corr_z

    prices = 100 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ══════════════════════════════════════════════════════════════════════════════
#  RETURN & RISK ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def annualised_stats(returns: pd.DataFrame):
    mu    = returns.mean() * TRADING_DAYS
    sigma = returns.std()  * np.sqrt(TRADING_DAYS)
    return mu, sigma


def portfolio_performance(weights, mu, cov):
    w   = np.array(weights)
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sr  = (ret - RISK_FREE) / vol
    return ret, vol, sr


def max_sharpe(mu, cov):
    n = len(mu)
    def neg_sr(w):
        r, v, _ = portfolio_performance(w, mu, cov)
        return -(r - RISK_FREE) / v

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = [(0, 1)] * n
    x0          = np.ones(n) / n
    res         = minimize(neg_sr, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def min_variance(mu, cov):
    n = len(mu)
    def port_vol(w):
        return np.sqrt(np.array(w) @ cov @ np.array(w))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds      = [(0, 1)] * n
    x0          = np.ones(n) / n
    res         = minimize(port_vol, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def equal_weight(n):
    return np.ones(n) / n


def efficient_frontier(mu, cov, n_points=60):
    """Return (vols, rets) along the efficient frontier."""
    r_min = mu.min()
    r_max = mu.max()
    targets = np.linspace(r_min, r_max, n_points)
    vols, rets = [], []
    n = len(mu)
    for target in targets:
        def port_vol(w):
            return np.sqrt(np.array(w) @ cov @ np.array(w))
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.array(w) @ mu - t},
        ]
        bounds = [(0, 1)] * n
        res = minimize(port_vol, np.ones(n)/n, method="SLSQP",
                       bounds=bounds, constraints=constraints)
        if res.success:
            vols.append(res.fun)
            rets.append(target)
    return np.array(vols), np.array(rets)


def monte_carlo_portfolios(mu, cov, n=N_PORTFOLIOS):
    rng = np.random.default_rng(0)
    n_assets = len(mu)
    results  = np.zeros((3, n))
    for i in range(n):
        w = rng.dirichlet(np.ones(n_assets))
        r, v, sr = portfolio_performance(w, mu, cov)
        results[:, i] = r, v, sr
    return results  # shape (3, N): ret, vol, sharpe


# ══════════════════════════════════════════════════════════════════════════════
#  RISK METRICS
# ══════════════════════════════════════════════════════════════════════════════

def drawdown_series(cum_ret: pd.Series) -> pd.Series:
    roll_max = cum_ret.cummax()
    return (cum_ret - roll_max) / roll_max


def max_drawdown(cum_ret: pd.Series) -> float:
    return drawdown_series(cum_ret).min()


def value_at_risk(returns: pd.Series, confidence=0.95) -> float:
    return np.percentile(returns, (1 - confidence) * 100)


def cvar(returns: pd.Series, confidence=0.95) -> float:
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()


def calmar_ratio(ann_ret: float, max_dd: float) -> float:
    return ann_ret / abs(max_dd) if max_dd != 0 else np.nan


def sortino_ratio(returns: pd.Series, ann_ret: float) -> float:
    neg  = returns[returns < 0]
    dsd  = neg.std() * np.sqrt(TRADING_DAYS)
    return (ann_ret - RISK_FREE) / dsd if dsd != 0 else np.nan


def beta_alpha(port_ret: pd.Series, bench_ret: pd.Series):
    cov_mat = np.cov(port_ret, bench_ret)
    beta    = cov_mat[0, 1] / cov_mat[1, 1]
    alpha   = (port_ret.mean() - beta * bench_ret.mean()) * TRADING_DAYS
    return alpha, beta


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLING METRICS
# ══════════════════════════════════════════════════════════════════════════════

def rolling_sharpe(returns: pd.Series, window=63) -> pd.Series:
    r = returns.rolling(window)
    return (r.mean() * TRADING_DAYS - RISK_FREE) / (r.std() * np.sqrt(TRADING_DAYS))


def rolling_vol(returns: pd.Series, window=21) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return (returns * weights).sum(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GREY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.title.set_color(WHITE)
    ax.title.set_fontsize(10)
    ax.title.set_fontweight("bold")
    if title:  ax.set_title(title, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=GREY, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=GREY, fontsize=8)
    ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)


def _pct_fmt(x, _): return f"{x:.0%}"
def _bps_fmt(x, _):  return f"{x*100:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def build_dashboard(prices: pd.DataFrame):
    # ── computations ──────────────────────────────────────────────────────────
    returns    = compute_returns(prices)
    mu, sigma  = annualised_stats(returns)
    cov        = returns.cov() * TRADING_DAYS

    w_ms  = max_sharpe(mu.values, cov.values)
    w_mv  = min_variance(mu.values, cov.values)
    w_ew  = equal_weight(len(mu))

    port_ms  = build_portfolio_returns(returns, w_ms)
    port_mv  = build_portfolio_returns(returns, w_mv)
    port_ew  = build_portfolio_returns(returns, w_ew)

    # benchmark = equal-weight for alpha/beta
    bench = port_ew

    cum_ms = (1 + port_ms).cumprod()
    cum_mv = (1 + port_mv).cumprod()
    cum_ew = (1 + port_ew).cumprod()

    mc_res       = monte_carlo_portfolios(mu.values, cov.values)
    ef_vols, ef_rets = efficient_frontier(mu.values, cov.values)

    r_ms, v_ms, sr_ms = portfolio_performance(w_ms, mu.values, cov.values)
    r_mv, v_mv, sr_mv = portfolio_performance(w_mv, mu.values, cov.values)
    r_ew, v_ew, sr_ew = portfolio_performance(w_ew, mu.values, cov.values)

    # risk metrics (max-sharpe portfolio)
    dd_ms  = drawdown_series(cum_ms)
    var95  = value_at_risk(port_ms, 0.95)
    cvar95 = cvar(port_ms, 0.95)
    mdd    = max_drawdown(cum_ms)
    calmar = calmar_ratio(r_ms, mdd)
    sortino = sortino_ratio(port_ms, r_ms)
    alpha_ms, beta_ms = beta_alpha(port_ms, bench)

    roll_sr  = rolling_sharpe(port_ms)
    roll_vol = rolling_vol(port_ms)

    # correlation matrix
    corr = returns.corr()

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 18), facecolor=BG)
    fig.subplots_adjust(hspace=0.42, wspace=0.32, top=0.93, bottom=0.04,
                        left=0.05, right=0.97)

    gs = gridspec.GridSpec(4, 4, figure=fig)

    # Header
    fig.text(0.5, 0.965, "QUANTITATIVE PORTFOLIO ANALYTICS  ·  CIO BRIEFING",
             ha="center", va="top", color=WHITE, fontsize=16, fontweight="bold",
             fontfamily="monospace")
    fig.text(0.5, 0.948,
             f"Modern Portfolio Theory  ·  {prices.index[0].date()} → {prices.index[-1].date()}"
             f"  ·  {len(mu)} Assets  ·  Risk-Free Rate {RISK_FREE:.0%}",
             ha="center", va="top", color=GREY, fontsize=9)

    # ── 1. Cumulative Returns ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _style_ax(ax1, "Cumulative Return  (Indexed to 100)")
    for cum, lbl, col in [(cum_ms, "Max-Sharpe", GOLD),
                          (cum_mv, "Min-Variance", TEAL),
                          (cum_ew, "Equal-Weight", LAVENDER)]:
        ax1.plot(cum.index, cum * 100, color=col, linewidth=1.6, label=lbl)
    ax1.legend(framealpha=0, labelcolor=WHITE, fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax1.set_ylabel("Portfolio Value", color=GREY, fontsize=8)

    # ── 2. Efficient Frontier ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    _style_ax(ax2, "Efficient Frontier  (Monte Carlo + Optimised Points)")
    sc = ax2.scatter(mc_res[1], mc_res[0], c=mc_res[2], cmap="plasma",
                     s=3, alpha=0.55, zorder=1)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=GREY, labelsize=7)
    cbar.set_label("Sharpe Ratio", color=GREY, fontsize=7)
    ax2.plot(ef_vols, ef_rets, color=WHITE, linewidth=2, zorder=3, label="Efficient Frontier")
    for (rv, rr, lbl, col, m) in [
        (v_ms, r_ms, f"Max-Sharpe  SR={sr_ms:.2f}", GOLD,    "*"),
        (v_mv, r_mv, f"Min-Var      SR={sr_mv:.2f}", TEAL,    "D"),
        (v_ew, r_ew, f"Equal-Wt    SR={sr_ew:.2f}", LAVENDER,"o"),
    ]:
        ax2.scatter(rv, rr, color=col, s=160, marker=m, zorder=5,
                    edgecolors="white", linewidths=0.8, label=lbl)
    ax2.legend(framealpha=0, labelcolor=WHITE, fontsize=7.5, loc="upper left")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax2.set_xlabel("Annualised Volatility", color=GREY, fontsize=8)
    ax2.set_ylabel("Annualised Return",     color=GREY, fontsize=8)

    # ── 3. Drawdown ───────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    _style_ax(ax3, "Drawdown  (Max-Sharpe Portfolio)")
    ax3.fill_between(dd_ms.index, dd_ms.values * 100, 0,
                     color=CRIMSON, alpha=0.45, linewidth=0)
    ax3.plot(dd_ms.index, dd_ms.values * 100, color=CRIMSON, linewidth=1)
    ax3.axhline(mdd * 100, color=ORANGE, linewidth=1, linestyle="--",
                label=f"Max DD  {mdd:.1%}")
    ax3.legend(framealpha=0, labelcolor=WHITE, fontsize=8)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax3.set_ylabel("Drawdown %", color=GREY, fontsize=8)

    # ── 4. Rolling Sharpe ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    _style_ax(ax4, "Rolling 63-Day Sharpe  (Max-Sharpe Portfolio)")
    ax4.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax4.axhline(1, color=GREEN, linewidth=0.7, linestyle=":", alpha=0.7)
    ax4.plot(roll_sr.index, roll_sr.values, color=TEAL, linewidth=1.3)
    ax4.fill_between(roll_sr.index, roll_sr.values, 0,
                     where=roll_sr > 0, color=GREEN, alpha=0.15, linewidth=0)
    ax4.fill_between(roll_sr.index, roll_sr.values, 0,
                     where=roll_sr < 0, color=CRIMSON, alpha=0.15, linewidth=0)
    ax4.set_ylabel("Sharpe", color=GREY, fontsize=8)

    # ── 5. Return Distribution + VaR/CVaR ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    _style_ax(ax5, "Daily Return Distribution  (Max-Sharpe)")
    n_bins = 80
    counts, bins, patches = ax5.hist(port_ms * 100, bins=n_bins,
                                     color=TEAL, alpha=0.6, density=True, edgecolor="none")
    # overlay normal
    x  = np.linspace(port_ms.min()*100, port_ms.max()*100, 300)
    ax5.plot(x, norm.pdf(x, port_ms.mean()*100, port_ms.std()*100),
             color=GOLD, linewidth=2, label="Normal Fit")
    ax5.axvline(var95 * 100, color=ORANGE, linewidth=1.5, linestyle="--",
                label=f"VaR 95%  {var95:.2%}")
    ax5.axvline(cvar95 * 100, color=CRIMSON, linewidth=1.5, linestyle="--",
                label=f"CVaR 95%  {cvar95:.2%}")
    ax5.legend(framealpha=0, labelcolor=WHITE, fontsize=7.5)
    ax5.set_xlabel("Daily Return %", color=GREY, fontsize=8)
    ax5.set_ylabel("Density", color=GREY, fontsize=8)

    # ── 6. Correlation Heatmap ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    _style_ax(ax6, "Asset Correlation Matrix")
    cmap_corr = LinearSegmentedColormap.from_list(
        "corr", [CRIMSON, PANEL, TEAL], N=256)
    im = ax6.imshow(corr.values, cmap=cmap_corr, vmin=-1, vmax=1, aspect="auto")
    tickers = list(corr.columns)
    ax6.set_xticks(range(len(tickers))); ax6.set_xticklabels(tickers, rotation=45, ha="right", fontsize=7, color=WHITE)
    ax6.set_yticks(range(len(tickers))); ax6.set_yticklabels(tickers, fontsize=7, color=WHITE)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val = corr.values[i, j]
            ax6.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=5.5, color=WHITE if abs(val) < 0.6 else BG)
    cb2 = fig.colorbar(im, ax=ax6, fraction=0.035, pad=0.02)
    cb2.ax.tick_params(colors=GREY, labelsize=7)

    # ── 7. Portfolio Weights (bar) ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, :2])
    _style_ax(ax7, "Portfolio Weights Comparison")
    tickers_list = list(returns.columns)
    x = np.arange(len(tickers_list))
    w = 0.26
    ax7.bar(x - w, w_ms * 100,  width=w, label="Max-Sharpe",  color=GOLD,    alpha=0.85)
    ax7.bar(x,     w_mv * 100,  width=w, label="Min-Variance", color=TEAL,    alpha=0.85)
    ax7.bar(x + w, w_ew * 100,  width=w, label="Equal-Wt",    color=LAVENDER, alpha=0.85)
    ax7.set_xticks(x); ax7.set_xticklabels(tickers_list, rotation=45, ha="right", fontsize=7, color=WHITE)
    ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax7.set_ylabel("Weight %", color=GREY, fontsize=8)
    ax7.legend(framealpha=0, labelcolor=WHITE, fontsize=8)

    # ── 8. Key Metrics Table ──────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 2:])
    ax8.set_facecolor(PANEL)
    for spine in ax8.spines.values():
        spine.set_edgecolor(BORDER)
    ax8.set_xticks([]); ax8.set_yticks([])
    ax8.set_title("Key Performance & Risk Metrics  (Max-Sharpe Portfolio)",
                  color=WHITE, fontsize=10, fontweight="bold", pad=8)

    # assemble metrics
    metrics = [
        ("Annualised Return",      f"{r_ms:.2%}"),
        ("Annualised Volatility",  f"{v_ms:.2%}"),
        ("Sharpe Ratio",           f"{sr_ms:.3f}"),
        ("Sortino Ratio",          f"{sortino:.3f}"),
        ("Calmar Ratio",           f"{calmar:.3f}"),
        ("Max Drawdown",           f"{mdd:.2%}"),
        ("VaR 95% (Daily)",        f"{var95:.2%}"),
        ("CVaR 95% (Daily)",       f"{cvar95:.2%}"),
        ("Skewness",               f"{skew(port_ms):.3f}"),
        ("Excess Kurtosis",        f"{kurtosis(port_ms):.3f}"),
        ("Alpha (ann.)",           f"{alpha_ms:.2%}"),
        ("Beta  (vs EW bench)",    f"{beta_ms:.3f}"),
    ]

    cols = 2
    rows_per_col = (len(metrics) + 1) // cols
    col_x = [0.04, 0.52]
    row_start_y = 0.88
    row_gap     = 0.072

    for idx, (label, value) in enumerate(metrics):
        col = idx // rows_per_col
        row = idx % rows_per_col
        x_pos = col_x[col]
        y_pos = row_start_y - row * row_gap

        # coloured background pill per value
        val_color = GREEN if "Return" in label or "Sharpe" in label or "Sortino" in label \
                    else (CRIMSON if "Drawdown" in label or "VaR" in label or "CVaR" in label \
                    else GOLD)

        ax8.text(x_pos, y_pos, label, transform=ax8.transAxes,
                 color=GREY, fontsize=8.5, va="center")
        ax8.text(x_pos + 0.42, y_pos, value, transform=ax8.transAxes,
                 color=val_color, fontsize=9, fontweight="bold", va="center")
        # horizontal rule (using transAxes line)
        line_y = y_pos - row_gap * 0.38
        ax8.plot([col * 0.5 + 0.02, (col + 1) * 0.5 - 0.02],
                 [line_y, line_y], color=BORDER, linewidth=0.5,
                 transform=ax8.transAxes, clip_on=False)

    # ── footer ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.012,
             "CONFIDENTIAL  ·  For Internal Use Only  ·  Past performance is not indicative of future results  ·  "
             "All figures assume long-only, fully-invested, no-leverage portfolios.",
             ha="center", color=GREY, fontsize=7, style="italic")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MPT CIO Dashboard")
    parser.add_argument("--csv", default=None,
                        help="Path to price CSV (tickers as columns, dates as index)")
    parser.add_argument("--out", default="mpt_cio_dashboard.png",
                        help="Output file path (default: mpt_cio_dashboard.png)")
    args = parser.parse_args()

    prices = load_prices(args.csv)
    print("[MPT] Computing returns and running optimisation …")
    fig    = build_dashboard(prices)

    out_path = args.out
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"[DONE] Dashboard saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()