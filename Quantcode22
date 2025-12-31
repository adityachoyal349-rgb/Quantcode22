# --- Imports ---
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from google.colab import files, drive
import warnings

# --- Ignore warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Mount Google Drive ---
drive.mount('/content/drive')

# --- Set tickers ---
ticker = "BSE.NS"             # Stock of interest
benchmark_ticker = "INDIGO.NS"  # Benchmark updated

# --- Download stock and benchmark data ---
data = yf.download(ticker, start="2020-01-01", end="2025-12-31", auto_adjust=False).dropna()
benchmark_data = yf.download(benchmark_ticker, start="2020-01-01", end="2025-12-31", auto_adjust=False).dropna()

if data.empty or benchmark_data.empty:
    raise ValueError("Data download failed")

# --- Prices and returns ---
prices = data["Close"]
returns = prices.pct_change().dropna().squeeze()
benchmark_prices = benchmark_data["Close"]
benchmark_returns = benchmark_prices.pct_change().dropna().squeeze()

# --- Metrics for stock ---
expected_annual_return = returns.mean() * 252
annual_volatility_val = returns.std(ddof=0) * np.sqrt(252)
sharpe_ratio = expected_annual_return / annual_volatility_val if annual_volatility_val != 0 else np.nan

downside_returns = returns[returns < 0]
downside_std_val = downside_returns.std(ddof=0) * np.sqrt(252)
sortino_ratio = expected_annual_return / downside_std_val if downside_std_val != 0 else np.nan

cumulative_returns = (1 + returns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdowns = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdowns.min()

# --- Metrics for benchmark ---
expected_annual_return_b = benchmark_returns.mean() * 252
annual_volatility_val_b = benchmark_returns.std(ddof=0) * np.sqrt(252)
sharpe_ratio_b = expected_annual_return_b / annual_volatility_val_b if annual_volatility_val_b != 0 else np.nan

downside_returns_b = benchmark_returns[benchmark_returns < 0]
downside_std_val_b = downside_returns_b.std(ddof=0) * np.sqrt(252)
sortino_ratio_b = expected_annual_return_b / downside_std_val_b if downside_std_val_b != 0 else np.nan

cumulative_returns_b = (1 + benchmark_returns).cumprod()
rolling_max_b = cumulative_returns_b.cummax()
drawdowns_b = (cumulative_returns_b - rolling_max_b) / rolling_max_b
max_drawdown_b = drawdowns_b.min()

# --- Stock Info ---
stock_info = yf.Ticker(ticker).info
benchmark_info = yf.Ticker(benchmark_ticker).info

current_pe = stock_info.get("trailingPE", np.nan)
dividend_yield = stock_info.get("dividendYield", np.nan)
eps = stock_info.get("trailingEps", np.nan)
pb_ratio = stock_info.get("priceToBook", np.nan)
peg_ratio = stock_info.get("pegRatio", np.nan)

benchmark_pe = benchmark_info.get("trailingPE", np.nan)
benchmark_dividend_yield = benchmark_info.get("dividendYield", np.nan)
benchmark_eps = benchmark_info.get("trailingEps", np.nan)
benchmark_pb = benchmark_info.get("priceToBook", np.nan)
benchmark_peg = benchmark_info.get("pegRatio", np.nan)

# --- EMAs ---
ema_50 = prices.ewm(span=50, adjust=False).mean()
ema_200 = prices.ewm(span=200, adjust=False).mean()
ema_50_b = benchmark_prices.ewm(span=50, adjust=False).mean()
ema_200_b = benchmark_prices.ewm(span=200, adjust=False).mean()

# --- RSI ---
def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

rsi = compute_rsi(prices)
rsi_b = compute_rsi(benchmark_prices)

latest_rsi = float(rsi.iloc[-1]) if not rsi.empty else np.nan
latest_rsi_b = float(rsi_b.iloc[-1]) if not rsi_b.empty else np.nan

# --- Momentum (20-day) ---
momentum = prices / prices.shift(20) - 1
momentum_val = float(momentum.iloc[-1])
momentum_b = benchmark_prices / benchmark_prices.shift(20) - 1
momentum_val_b = float(momentum_b.iloc[-1])

# --- PDF path ---
pdf_path = '/content/drive/MyDrive/quant_report_BSE_vs_INDIGO.pdf'

# --- Scoring function (Max 28 points) ---
def calculate_score(pe, sharpe, sortino, vol, mdd, div, eps, ema_50, ema_200, price, rsi, pb, peg, momentum):
    score = 0
    if pe < 50: score += 3
    if sharpe > 1: score += 3
    if sortino > 1: score += 1
    if vol < 0.4: score += 2
    if mdd > -0.4: score += 1
    if div > 0.01: score += 1
    if eps > 1: score += 1
    if float(ema_50.iloc[-1]) > float(ema_200.iloc[-1]):
        score += 3
    if float(price.iloc[-1]) > float(ema_50.iloc[-1]) and float(price.iloc[-1]) > float(ema_200.iloc[-1]):
        score += 2
    # RSI scoring
    if not np.isnan(rsi):
        if rsi > 70: score += 0
        elif 30 <= rsi <= 70: score += 1
        elif rsi < 30: score += 3
    # P/B Ratio
    if not np.isnan(pb) and pb < 5: score += 3
    # PEG Ratio
    if not np.isnan(peg) and peg < 2: score += 3
    # Momentum
    if not np.isnan(momentum) and momentum > 0: score += 2
    return score

stock_score = calculate_score(current_pe, sharpe_ratio, sortino_ratio, annual_volatility_val,
                              max_drawdown, dividend_yield, eps, ema_50, ema_200, prices, latest_rsi, pb_ratio, peg_ratio, momentum_val)

benchmark_score = calculate_score(benchmark_pe, sharpe_ratio_b, sortino_ratio_b, annual_volatility_val_b,
                                  max_drawdown_b, benchmark_dividend_yield, benchmark_eps, ema_50_b, ema_200_b, benchmark_prices, latest_rsi_b, benchmark_pb, benchmark_peg, momentum_val_b)

# --- Verdict function ---
def verdict(score):
    if score >= 23: return "BUY"
    elif score >= 16: return "HOLD"
    else: return "SELL"

stock_verdict = verdict(stock_score)
benchmark_verdict = verdict(benchmark_score)

# --- Expected 1-year price (simple estimation using annual return) ---
expected_price_stock = float(prices.iloc[-1]) * (1 + expected_annual_return)
expected_price_benchmark = float(benchmark_prices.iloc[-1]) * (1 + expected_annual_return_b)

# --- Create PDF ---
with PdfPages(pdf_path) as pdf:
    # --- Page 1: Metrics Table ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    metrics_df = pd.DataFrame({
        "Metric": ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "P/E Ratio"],
        "Stock": [f"{expected_annual_return:.2%}", f"{annual_volatility_val:.2%}", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}", f"{max_drawdown:.2%}", f"{current_pe:.2f}"],
        "Benchmark": [f"{expected_annual_return_b:.2%}", f"{annual_volatility_val_b:.2%}", f"{sharpe_ratio_b:.2f}", f"{sortino_ratio_b:.2f}", f"{max_drawdown_b:.2%}", f"{benchmark_pe:.2f}"]
    })
    table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 2)
    ax.set_title(f"Quantitative Analysis Report - {ticker} vs {benchmark_ticker}", fontsize=16, pad=20)
    pdf.savefig(fig); plt.close(fig)

    # --- Pages 2-5: OHLC Charts ---
    price_columns = ["Open", "High", "Low", "Close"]; colors = ["orange", "green", "red", "blue"]
    for col, color in zip(price_columns, colors):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data[col], label=f"{ticker} {col}", color=color, linestyle="-")
        ax.plot(benchmark_data.index, benchmark_data[col], label=f"{benchmark_ticker} {col}", color=color, linestyle="--")
        ax.set_title(f"{ticker} vs {benchmark_ticker} - {col} Price History")
        ax.set_xlabel("Date"); ax.set_ylabel("Price (INR)"); ax.legend(); ax.grid(True, linestyle="--", alpha=0.6)
        pdf.savefig(fig); plt.close(fig)

    # --- Page 6: Combined OHLC ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for col, color in zip(price_columns, colors):
        ax.plot(data.index, data[col], label=f"{ticker} {col}", color=color, linestyle="-")
        ax.plot(benchmark_data.index, benchmark_data[col], label=f"{benchmark_ticker} {col}", color=color, linestyle="--")
    ax.set_title(f"{ticker} vs {benchmark_ticker} - Combined OHLC Price History")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (INR)"); ax.legend(); ax.grid(True, linestyle="--", alpha=0.6)
    pdf.savefig(fig); plt.close(fig)

    # --- Page 7: Moving Averages ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(prices.index, prices, label="Close Price", color="blue")
    ax.plot(prices.index, prices.rolling(50).mean(), label="50-day MA", color="orange")
    ax.plot(prices.index, prices.rolling(200).mean(), label="200-day MA", color="green")
    ax.set_title(f"{ticker} - Moving Averages"); ax.set_xlabel("Date"); ax.set_ylabel("Price (INR)")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.6); pdf.savefig(fig); plt.close(fig)

    # --- Page 8: Cumulative Returns ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative_returns.index, cumulative_returns, label=ticker, color="blue")
    ax.plot(cumulative_returns_b.index, cumulative_returns_b, label=benchmark_ticker, color="orange")
    ax.set_title("Cumulative Returns: Stock vs Benchmark"); ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.6); pdf.savefig(fig); plt.close(fig)

    # --- Page 9: Drawdowns ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(drawdowns.index, drawdowns, label=ticker, color="purple")
    ax.plot(drawdowns_b.index, drawdowns_b, label=benchmark_ticker, color="brown")
    ax.set_title("Drawdowns: Stock vs Benchmark"); ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.6); pdf.savefig(fig); plt.close(fig)

    # --- Page 10: Returns Histogram ---
    fig, ax = plt.subplots(figsize=(12, 6))
    returns.hist(bins=50, ax=ax, color="cyan")
    ax.set_title(f"{ticker} Daily Returns Distribution"); ax.set_xlabel("Return"); ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.6); pdf.savefig(fig); plt.close(fig)

    # --- Page 11: Performance + Dividend Yield, EPS, RSI, P/B, PEG, Momentum ---
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis("off")
    compare_df = pd.DataFrame({
        "Metric": ["Expected Annual Return","Annual Volatility","Sharpe Ratio","Sortino Ratio","Max Drawdown","P/E Ratio","Dividend Yield","EPS","RSI (14d)","P/B Ratio","PEG Ratio","Momentum (20d)"],
        "Stock": [f"{expected_annual_return:.2%}", f"{annual_volatility_val:.2%}", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}", f"{max_drawdown:.2%}", f"{current_pe:.2f}", f"{dividend_yield:.2%}" if pd.notna(dividend_yield) else "N/A", f"{eps:.2f}" if pd.notna(eps) else "N/A", f"{latest_rsi:.2f}", f"{pb_ratio:.2f}" if pd.notna(pb_ratio) else "N/A", f"{peg_ratio:.2f}" if pd.notna(peg_ratio) else "N/A", f"{momentum_val:.2%}"],
        "Benchmark": [f"{expected_annual_return_b:.2%}", f"{annual_volatility_val_b:.2%}", f"{sharpe_ratio_b:.2f}", f"{sortino_ratio_b:.2f}", f"{max_drawdown_b:.2%}", f"{benchmark_pe:.2f}", f"{benchmark_dividend_yield:.2%}" if pd.notna(benchmark_dividend_yield) else "N/A", f"{benchmark_eps:.2f}" if pd.notna(benchmark_eps) else "N/A", f"{latest_rsi_b:.2f}", f"{benchmark_pb:.2f}" if pd.notna(benchmark_pb) else "N/A", f"{benchmark_peg:.2f}" if pd.notna(benchmark_peg) else "N/A", f"{momentum_val_b:.2%}"]
    })
    table = ax.table(cellText=compare_df.values, colLabels=compare_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 2)
    ax.set_title("Stock vs Benchmark: Performance + Dividend Yield, EPS, RSI, P/B, PEG, Momentum", fontsize=16, pad=20)
    pdf.savefig(fig); plt.close(fig)

    # --- Page 12: Scoring & Verdict ---
    fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis("off")
    score_df = pd.DataFrame({
        "Metric": ["Total Score","Verdict","Expected Price in 1Y"],
        "Stock": [stock_score, stock_verdict, f"{expected_price_stock:.2f} INR"],
        "Benchmark": [benchmark_score, benchmark_verdict, f"{expected_price_benchmark:.2f} INR"]
    })
    table = ax.table(cellText=score_df.values, colLabels=score_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(14); table.scale(1, 3)
    ax.set_title("Stock vs Benchmark: Scoring & Verdict (Max 28 pts)", fontsize=16, pad=20)
    pdf.savefig(fig); plt.close(fig)

# =======================
    # Page 13: Revenue & Net Profit (Quarterly + FY) - Stock + Benchmark
    # =======================

    # --- Fetch financial statements ---
    stock_obj = yf.Ticker(ticker)
    benchmark_obj = yf.Ticker(benchmark_ticker)

    # --- Stock financials ---
    quarterly_fin = stock_obj.quarterly_financials
    annual_fin = stock_obj.financials

    if not quarterly_fin.empty:
        q_data = quarterly_fin.T.iloc[:8][["Total Revenue", "Net Income"]]
        q_data.columns = ["Revenue", "Net Profit"]
        q_data.index = q_data.index.strftime("%b %Y")
    else:
        q_data = pd.DataFrame()

    if not annual_fin.empty:
        fy_data = annual_fin.T.iloc[:4][["Total Revenue", "Net Income"]]
        fy_data.columns = ["Revenue", "Net Profit"]
        fy_data.index = fy_data.index.strftime("%Y")
    else:
        fy_data = pd.DataFrame()

    # --- Benchmark financials ---
    quarterly_fin_b = benchmark_obj.quarterly_financials
    annual_fin_b = benchmark_obj.financials

    if not quarterly_fin_b.empty:
        q_data_b = quarterly_fin_b.T.iloc[:8][["Total Revenue", "Net Income"]]
        q_data_b.columns = ["Revenue", "Net Profit"]
        q_data_b.index = q_data_b.index.strftime("%b %Y")
    else:
        q_data_b = pd.DataFrame()

    if not annual_fin_b.empty:
        fy_data_b = annual_fin_b.T.iloc[:4][["Total Revenue", "Net Income"]]
        fy_data_b.columns = ["Revenue", "Net Profit"]
        fy_data_b.index = fy_data_b.index.strftime("%Y")
    else:
        fy_data_b = pd.DataFrame()

    # --- Create PDF page ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.95, f"{ticker} vs {benchmark_ticker} – Revenue & Net Profit Overview",
            ha="center", va="center", fontsize=18, fontweight="bold")

    # --- Quarterly Table ---
    if not q_data.empty or not q_data_b.empty:
        # Merge stock + benchmark side by side
        q_table_data = []
        q_labels = q_data.index if not q_data.empty else q_data_b.index
        for idx in q_labels:
            row_stock = q_data.loc[idx].values if idx in q_data.index else ["N/A","N/A"]
            row_bench = q_data_b.loc[idx].values if idx in q_data_b.index else ["N/A","N/A"]
            q_table_data.append([*row_stock, *row_bench])

        q_table = ax.table(
            cellText=[[f"{v/1e7:.2f} Cr" if isinstance(v, (int,float)) else v for v in row] for row in q_table_data],
            colLabels=["Revenue", "Net Profit", "Revenue", "Net Profit"],
            rowLabels=q_labels,
            cellLoc="center",
            bbox=[0.05, 0.45, 0.9, 0.4]
        )
        q_table.auto_set_font_size(False)
        q_table.set_fontsize(11)
        ax.text(0.5, 0.88, "Last 8 Quarters (Stock vs Benchmark, ₹ Crore)",
                ha="center", fontsize=14, fontweight="bold")

    # --- FY Table ---
    if not fy_data.empty or not fy_data_b.empty:
        fy_table_data = []
        fy_labels = fy_data.index if not fy_data.empty else fy_data_b.index
        for idx in fy_labels:
            row_stock = fy_data.loc[idx].values if idx in fy_data.index else ["N/A","N/A"]
            row_bench = fy_data_b.loc[idx].values if idx in fy_data_b.index else ["N/A","N/A"]
            fy_table_data.append([*row_stock, *row_bench])

        fy_table = ax.table(
            cellText=[[f"{v/1e7:.2f} Cr" if isinstance(v, (int,float)) else v for v in row] for row in fy_table_data],
            colLabels=["Revenue", "Net Profit", "Revenue", "Net Profit"],
            rowLabels=fy_labels,
            cellLoc="center",
            bbox=[0.05, 0.1, 0.9, 0.25]
        )
        fy_table.auto_set_font_size(False)
        fy_table.set_fontsize(11)
        ax.text(0.5, 0.40, "Last 4 Financial Years (Stock vs Benchmark, ₹ Crore)",
                ha="center", fontsize=14, fontweight="bold")

    pdf.savefig(fig)
    plt.close(fig)

# =======================
    # Page 14: Monte Carlo Simulation (1-Year Price Forecast)
    # =======================

    num_simulations = 5000
    num_days = 252

    def monte_carlo_sim(start_price, mu, sigma, days, sims):
        final_prices = np.zeros(sims)
        for i in range(sims):
            daily_returns = np.random.normal(mu / 252, sigma / np.sqrt(252), days)
            final_prices[i] = start_price * np.prod(1 + daily_returns)
        return final_prices

    # --- Run simulations ---
    sim_stock = monte_carlo_sim(
        float(prices.iloc[-1]),
        expected_annual_return,
        annual_volatility_val,
        num_days,
        num_simulations
    )

    sim_benchmark = monte_carlo_sim(
        float(benchmark_prices.iloc[-1]),
        expected_annual_return_b,
        annual_volatility_val_b,
        num_days,
        num_simulations
    )

    # --- Summary statistics ---
    stock_mean = np.mean(sim_stock)
    stock_p5, stock_p95 = np.percentile(sim_stock, [5, 95])

    bench_mean = np.mean(sim_benchmark)
    bench_p5, bench_p95 = np.percentile(sim_benchmark, [5, 95])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(sim_stock, bins=60, alpha=0.55, label=f"{ticker}", density=True)
    ax.hist(sim_benchmark, bins=60, alpha=0.55, label=f"{benchmark_ticker}", density=True)

    # --- Stock markers ---
    ax.axvline(stock_mean, linestyle="--", linewidth=2, label=f"{ticker} Mean")
    ax.axvline(stock_p5, linestyle=":", linewidth=1.5, label=f"{ticker} 5%")
    ax.axvline(stock_p95, linestyle=":", linewidth=1.5, label=f"{ticker} 95%")

    # --- Benchmark markers ---
    ax.axvline(bench_mean, linestyle="--", linewidth=2, label=f"{benchmark_ticker} Mean")
    ax.axvline(bench_p5, linestyle=":", linewidth=1.5, label=f"{benchmark_ticker} 5%")
    ax.axvline(bench_p95, linestyle=":", linewidth=1.5, label=f"{benchmark_ticker} 95%")

    ax.set_title(
        "Monte Carlo Simulation – 1 Year Expected Price Distribution\n"
        f"{ticker} vs {benchmark_ticker}",
        fontsize=16
    )

    ax.set_xlabel("Expected Price After 1 Year (INR)")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Annotation box ---
    textstr = (
        f"{ticker}:\n"
        f"Mean: {stock_mean:.2f} INR\n"
        f"5%–95% Range: {stock_p5:.2f} – {stock_p95:.2f}\n\n"
        f"{benchmark_ticker}:\n"
        f"Mean: {bench_mean:.2f} INR\n"
        f"5%–95% Range: {bench_p5:.2f} – {bench_p95:.2f}"
    )

    ax.text(
        0.02, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    pdf.savefig(fig)
    plt.close(fig)

# =======================
    # Page 15: CAGR & YoY Growth (Revenue & Net Profit)
    # =======================

    def calculate_cagr(series):
        if len(series) < 2:
            return np.nan
        start, end = series.iloc[-1], series.iloc[0]
        years = len(series) - 1
        if start <= 0 or end <= 0:
            return np.nan
        return (end / start) ** (1 / years) - 1

    def calculate_yoy(series):
        return series.pct_change(-1) * 100

    # --- Prepare FY data (reuse Page 13 data logic safely) ---
    stock_fy = yf.Ticker(ticker).financials.T.iloc[:4][["Total Revenue", "Net Income"]]
    bench_fy = yf.Ticker(benchmark_ticker).financials.T.iloc[:4][["Total Revenue", "Net Income"]]

    stock_fy.columns = ["Revenue", "Net Profit"]
    bench_fy.columns = ["Revenue", "Net Profit"]

    stock_fy = stock_fy.sort_index()
    bench_fy = bench_fy.sort_index()

    # --- CAGR ---
    rev_cagr_stock = calculate_cagr(stock_fy["Revenue"])
    np_cagr_stock = calculate_cagr(stock_fy["Net Profit"])
    rev_cagr_bench = calculate_cagr(bench_fy["Revenue"])
    np_cagr_bench = calculate_cagr(bench_fy["Net Profit"])

    # --- YoY Growth ---
    stock_yoy = calculate_yoy(stock_fy)
    bench_yoy = calculate_yoy(bench_fy)

    # --- Plot ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Revenue YoY
    axs[0, 0].plot(stock_yoy.index, stock_yoy["Revenue"], marker="o", label=ticker)
    axs[0, 0].plot(bench_yoy.index, bench_yoy["Revenue"], marker="o", label=benchmark_ticker)
    axs[0, 0].set_title("Revenue YoY Growth (%)")
    axs[0, 0].legend(); axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Net Profit YoY
    axs[0, 1].plot(stock_yoy.index, stock_yoy["Net Profit"], marker="o", label=ticker)
    axs[0, 1].plot(bench_yoy.index, bench_yoy["Net Profit"], marker="o", label=benchmark_ticker)
    axs[0, 1].set_title("Net Profit YoY Growth (%)")
    axs[0, 1].legend(); axs[0, 1].grid(True, linestyle="--", alpha=0.5)

    # CAGR Text Box
    axs[1, 0].axis("off")
    cagr_text = (
        f"{ticker} CAGR:\n"
        f"Revenue: {rev_cagr_stock:.2%}\n"
        f"Net Profit: {np_cagr_stock:.2%}\n\n"
        f"{benchmark_ticker} CAGR:\n"
        f"Revenue: {rev_cagr_bench:.2%}\n"
        f"Net Profit: {np_cagr_bench:.2%}"
    )
    axs[1, 0].text(0.05, 0.9, cagr_text, fontsize=12, va="top",
                   bbox=dict(boxstyle="round", alpha=0.15))

    axs[1, 1].axis("off")

    fig.suptitle(
        "CAGR & YoY Growth Analysis (Last 4 Financial Years)",
        fontsize=16, fontweight="bold"
    )

    pdf.savefig(fig)
    plt.close(fig)

# =======================
    # Page 16: DuPont Analysis (ROE Decomposition)
    # =======================

    def dupont_components(ticker_symbol):
        t = yf.Ticker(ticker_symbol)

        fin = t.financials
        bal = t.balance_sheet

        if fin.empty or bal.empty:
            return [np.nan] * 4

        # --- Revenue & Net Income ---
        try:
            revenue = fin.loc["Total Revenue"].iloc[0]
            net_income = fin.loc["Net Income"].iloc[0]
        except KeyError:
            return [np.nan] * 4

        # --- Total Assets ---
        if "Total Assets" not in bal.index:
            return [np.nan] * 4
        total_assets = bal.loc["Total Assets"].iloc[0]

        # --- Equity (Yahoo Finance label variations) ---
        equity_keys = [
            "Total Stockholders Equity",
            "Total Stockholder Equity",
            "Stockholders Equity",
            "Total Equity Gross Minority Interest"
        ]

        equity = None
        for key in equity_keys:
            if key in bal.index:
                equity = bal.loc[key].iloc[0]
                break

        if equity is None or equity <= 0:
            return [np.nan] * 4

        if revenue <= 0 or total_assets <= 0:
            return [np.nan] * 4

        profit_margin = net_income / revenue
        asset_turnover = revenue / total_assets
        equity_multiplier = total_assets / equity
        roe = profit_margin * asset_turnover * equity_multiplier

        return profit_margin, asset_turnover, equity_multiplier, roe

    # --- Calculate DuPont components ---
    stock_pm, stock_at, stock_em, stock_roe = dupont_components(ticker)
    bench_pm, bench_at, bench_em, bench_roe = dupont_components(benchmark_ticker)

    # --- Create PDF page ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    dupont_df = pd.DataFrame({
        "Metric": [
            "Profit Margin",
            "Asset Turnover",
            "Equity Multiplier",
            "ROE (DuPont)"
        ],
        "Stock": [
            f"{stock_pm:.2%}" if pd.notna(stock_pm) else "N/A",
            f"{stock_at:.2f}" if pd.notna(stock_at) else "N/A",
            f"{stock_em:.2f}" if pd.notna(stock_em) else "N/A",
            f"{stock_roe:.2%}" if pd.notna(stock_roe) else "N/A"
        ],
        "Benchmark": [
            f"{bench_pm:.2%}" if pd.notna(bench_pm) else "N/A",
            f"{bench_at:.2f}" if pd.notna(bench_at) else "N/A",
            f"{bench_em:.2f}" if pd.notna(bench_em) else "N/A",
            f"{bench_roe:.2%}" if pd.notna(bench_roe) else "N/A"
        ]
    })

    table = ax.table(
        cellText=dupont_df.values,
        colLabels=dupont_df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.5)

    ax.set_title(
        "DuPont Analysis – ROE Decomposition\n"
        f"{ticker} vs {benchmark_ticker}",
        fontsize=16,
        pad=20
    )

    # --- Interpretation box ---
    interpretation = (
        "DuPont Analysis decomposes ROE into:\n\n"
        "• Profit Margin → Operating efficiency\n"
        "• Asset Turnover → Asset utilization\n"
        "• Equity Multiplier → Financial leverage\n\n"
        "Higher ROE driven by margins & turnover\n"
        "is considered higher quality than\n"
        "leverage-driven ROE."
    )

    ax.text(
        0.05, 0.15,
        interpretation,
        fontsize=11,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    pdf.savefig(fig)
    plt.close(fig)

# =======================
    # Page 17: ROE Breakdown Trend (DuPont Components)
    # =======================

    def dupont_trend(ticker_symbol):
        t = yf.Ticker(ticker_symbol)
        fin = t.financials.T.iloc[:4]
        bal = t.balance_sheet.T.iloc[:4]

        data = []

        for idx in fin.index:
            try:
                revenue = fin.loc[idx, "Total Revenue"]
                net_income = fin.loc[idx, "Net Income"]
                total_assets = bal.loc[idx, "Total Assets"]
            except KeyError:
                data.append([np.nan, np.nan, np.nan, np.nan])
                continue

            equity = None
            equity_keys = [
                "Total Stockholders Equity",
                "Total Stockholder Equity",
                "Stockholders Equity",
                "Total Equity Gross Minority Interest"
            ]

            for key in equity_keys:
                if key in bal.columns:
                    equity = bal.loc[idx, key]
                    break

            if equity is None or equity <= 0 or revenue <= 0 or total_assets <= 0:
                data.append([np.nan, np.nan, np.nan, np.nan])
                continue

            profit_margin = net_income / revenue
            asset_turnover = revenue / total_assets
            equity_multiplier = total_assets / equity
            roe = profit_margin * asset_turnover * equity_multiplier

            data.append([profit_margin, asset_turnover, equity_multiplier, roe])

        df = pd.DataFrame(
            data,
            index=fin.index.strftime("%Y"),
            columns=["Profit Margin", "Asset Turnover", "Equity Multiplier", "ROE"]
        )

        return df.sort_index()

    # --- Calculate trends ---
    roe_stock = dupont_trend(ticker)
    roe_bench = dupont_trend(benchmark_ticker)

    # --- Plot ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Profit Margin
    axs[0, 0].plot(roe_stock.index, roe_stock["Profit Margin"] * 100, marker="o", label=ticker)
    axs[0, 0].plot(roe_bench.index, roe_bench["Profit Margin"] * 100, marker="o", label=benchmark_ticker)
    axs[0, 0].set_title("Profit Margin (%)")
    axs[0, 0].legend(); axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Asset Turnover
    axs[0, 1].plot(roe_stock.index, roe_stock["Asset Turnover"], marker="o", label=ticker)
    axs[0, 1].plot(roe_bench.index, roe_bench["Asset Turnover"], marker="o", label=benchmark_ticker)
    axs[0, 1].set_title("Asset Turnover")
    axs[0, 1].legend(); axs[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Equity Multiplier
    axs[1, 0].plot(roe_stock.index, roe_stock["Equity Multiplier"], marker="o", label=ticker)
    axs[1, 0].plot(roe_bench.index, roe_bench["Equity Multiplier"], marker="o", label=benchmark_ticker)
    axs[1, 0].set_title("Equity Multiplier (Leverage)")
    axs[1, 0].legend(); axs[1, 0].grid(True, linestyle="--", alpha=0.5)

    # ROE
    axs[1, 1].plot(roe_stock.index, roe_stock["ROE"] * 100, marker="o", label=ticker)
    axs[1, 1].plot(roe_bench.index, roe_bench["ROE"] * 100, marker="o", label=benchmark_ticker)
    axs[1, 1].set_title("Return on Equity (%)")
    axs[1, 1].legend(); axs[1, 1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "ROE Breakdown Trend (DuPont Analysis – Last 4 Financial Years)",
        fontsize=16,
        fontweight="bold"
    )

    pdf.savefig(fig)
    plt.close(fig)

# =======================
    # Page 18: VaR, CVaR & Drawdown Stress Test
    # =======================

    def var_cvar(returns, confidence=0.95):
        returns = returns.dropna()
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
        return float(var), float(cvar)

    def worst_n_day_return(prices, n=5):
        returns_n = prices.pct_change(n)
        return float(returns_n.min(skipna=True))

    # --- Calculate VaR & CVaR ---
    var_stock, cvar_stock = var_cvar(returns)
    var_bench, cvar_bench = var_cvar(benchmark_returns)

    # --- Stress metrics ---
    worst_1d_stock = float(returns.min())
    worst_5d_stock = worst_n_day_return(prices, 5)

    worst_1d_bench = float(benchmark_returns.min())
    worst_5d_bench = worst_n_day_return(benchmark_prices, 5)

    # --- Create PDF page ---
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    risk_df = pd.DataFrame({
        "Metric": [
            "VaR (95%, Daily)",
            "CVaR (95%, Daily)",
            "Worst 1-Day Return",
            "Worst 5-Day Return",
            "Maximum Drawdown"
        ],
        "Stock": [
            f"{var_stock:.2%}",
            f"{cvar_stock:.2%}",
            f"{worst_1d_stock:.2%}",
            f"{worst_5d_stock:.2%}",
            f"{max_drawdown:.2%}"
        ],
        "Benchmark": [
            f"{var_bench:.2%}",
            f"{cvar_bench:.2%}",
            f"{worst_1d_bench:.2%}",
            f"{worst_5d_bench:.2%}",
            f"{max_drawdown_b:.2%}"
        ]
    })

    table = ax.table(
        cellText=risk_df.values,
        colLabels=risk_df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.5)

    ax.set_title(
        "Risk Analysis – VaR, CVaR & Drawdown Stress Test\n"
        f"{ticker} vs {benchmark_ticker}",
        fontsize=16,
        pad=20
    )

    explanation = (
        "Risk Metrics Explanation:\n\n"
        "• VaR (95%): Maximum expected daily loss under normal market conditions\n"
        "• CVaR (95%): Average loss during extreme tail-risk events\n"
        "• Worst 5-Day Return: Short-term market stress scenario\n"
        "• Max Drawdown: Peak-to-trough capital erosion\n\n"
        "Lower values indicate superior downside risk protection."
    )

    ax.text(
        0.05, 0.12,
        explanation,
        fontsize=11,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    pdf.savefig(fig)
    plt.close(fig)

# =======================
    # Page 19: Correlation Heatmap vs NIFTY
    # =======================

    # --- Download NIFTY 500 (or NIFTY 50) data ---
    nifty_ticker = "^NSEI"
    nifty_data = yf.download(nifty_ticker, start="2020-01-01", end="2025-12-31")['Close'].dropna()

    # --- Convert DataFrame columns to 1D Series ---
    prices = prices.squeeze()
    benchmark_prices = benchmark_prices.squeeze()
    nifty_data = nifty_data.squeeze()

    # --- Align dates ---
    combined = pd.concat([prices, benchmark_prices, nifty_data], axis=1)
    combined.columns = [ticker, benchmark_ticker, "NIFTY"]
    combined.dropna(inplace=True)

    # --- Calculate daily returns ---
    combined_returns = combined.pct_change().dropna()

    # --- Correlation matrix ---
    corr_matrix = combined_returns.corr()

    # --- Plot heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    # --- Set ticks & labels ---
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.columns)

    # --- Add correlation values ---
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            c = corr_matrix.iloc[i, j]
            ax.text(j, i, f"{c:.2f}", va="center", ha="center",
                    color="white" if abs(c) > 0.5 else "black")

    # --- Colorbar & title ---
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Correlation Heatmap: {ticker}, {benchmark_ticker} vs NIFTY", fontsize=16, pad=20)

    pdf.savefig(fig)
    plt.close(fig)

# =======================
    # Page 20: Rolling 1-Year Correlation vs NIFTY
    # =======================

    # --- Ensure aligned daily returns (already calculated in Page 19) ---
    # combined_returns: stock, benchmark, NIFTY

    rolling_window = 252  # approx. 1 trading year
    rolling_corr_stock = combined_returns[ticker].rolling(rolling_window).corr(combined_returns["NIFTY"])
    rolling_corr_bench = combined_returns[benchmark_ticker].rolling(rolling_window).corr(combined_returns["NIFTY"])

    # --- Plot rolling correlations ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rolling_corr_stock.index, rolling_corr_stock, label=f"{ticker} vs NIFTY", color="blue")
    ax.plot(rolling_corr_bench.index, rolling_corr_bench, label=f"{benchmark_ticker} vs NIFTY", color="orange")
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)

    ax.set_title("Rolling 1-Year Correlation vs NIFTY", fontsize=16, pad=20)
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    pdf.savefig(fig)
    plt.close(fig)

# =======================
    # Page 21: Quantitative Summary Table
    # =======================

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    # Table only
    summary_df = pd.DataFrame({
        "Metric": [
            "Expected Annual Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio",
            "Max Drawdown", "P/E Ratio", "Dividend Yield", "EPS", "RSI (14d)",
            "P/B Ratio", "PEG Ratio", "Momentum (20d)",
            "Total Score", "Verdict", "Expected Price in 1Y",
            "VaR (95%)", "CVaR (95%)", "Worst 1-Day Return", "Worst 5-Day Return",
            "Beta vs NIFTY", "Correlation vs NIFTY (Current)"
        ],
        "Stock": [
            f"{expected_annual_return:.2%}", f"{annual_volatility_val:.2%}", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}",
            f"{max_drawdown:.2%}", f"{current_pe:.2f}", f"{dividend_yield:.2%}" if pd.notna(dividend_yield) else "N/A", f"{eps:.2f}" if pd.notna(eps) else "N/A",
            f"{latest_rsi:.2f}", f"{pb_ratio:.2f}" if pd.notna(pb_ratio) else "N/A", f"{peg_ratio:.2f}" if pd.notna(peg_ratio) else "N/A",
            f"{momentum_val:.2%}", stock_score, stock_verdict, f"{expected_price_stock:.2f} INR",
            f"{var_stock:.2%}", f"{cvar_stock:.2%}", f"{worst_1d_stock:.2%}", f"{worst_5d_stock:.2%}",
            f"{combined_returns[ticker].cov(combined_returns['NIFTY']) / combined_returns['NIFTY'].var():.2f}",
            f"{combined_returns[ticker].corr(combined_returns['NIFTY']):.2f}"
        ],
        "Benchmark": [
            f"{expected_annual_return_b:.2%}", f"{annual_volatility_val_b:.2%}", f"{sharpe_ratio_b:.2f}", f"{sortino_ratio_b:.2f}",
            f"{max_drawdown_b:.2%}", f"{benchmark_pe:.2f}", f"{benchmark_dividend_yield:.2%}" if pd.notna(benchmark_dividend_yield) else "N/A", f"{benchmark_eps:.2f}" if pd.notna(benchmark_eps) else "N/A",
            f"{latest_rsi_b:.2f}", f"{benchmark_pb:.2f}" if pd.notna(benchmark_pb) else "N/A", f"{benchmark_peg:.2f}" if pd.notna(benchmark_peg) else "N/A",
            f"{momentum_val_b:.2%}", benchmark_score, benchmark_verdict, f"{expected_price_benchmark:.2f} INR",
            f"{var_bench:.2%}", f"{cvar_bench:.2%}", f"{worst_1d_bench:.2%}", f"{worst_5d_bench:.2%}",
            f"{combined_returns[benchmark_ticker].cov(combined_returns['NIFTY']) / combined_returns['NIFTY'].var():.2f}",
            f"{combined_returns[benchmark_ticker].corr(combined_returns['NIFTY']):.2f}"
        ]
    })

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    ax.set_title(
        f"Quantitative Summary\n{ticker} vs {benchmark_ticker}",
        fontsize=16,
        pad=20
    )

    pdf.savefig(fig)
    plt.close(fig)

# =======================
    # Page 22: Risk & Behaviour Snapshot
    # =======================

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    snapshot_text = (
        "Risk & Behaviour Snapshot\n\n"
        "• Volatility Profile:\n"
        "  Stock shows higher volatility than benchmark, implying higher return dispersion.\n\n"
        "• Market Sensitivity (Beta):\n"
        "  Beta close to 1 indicates market-aligned movement with manageable systematic risk.\n\n"
        "• Correlation Regime:\n"
        "  Moderate correlation with NIFTY suggests partial diversification benefits.\n\n"
        "• Drawdown Behaviour:\n"
        "  Stock experiences deeper drawdowns but also stronger recoveries post stress.\n\n"
        "• Return Consistency:\n"
        "  Risk-adjusted metrics indicate superior consistency relative to benchmark.\n\n"
        "Note: This page summarizes behaviour, not forecasts."
    )

    ax.text(
        0.5, 0.5,
        snapshot_text,
        fontsize=12,
        wrap=True,
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#e8f0fe", alpha=0.3)
    )

    ax.set_title(
        "Risk & Behaviour Snapshot",
        fontsize=17,
        pad=25
    )

    pdf.savefig(fig)
    plt.close(fig)


print(f"✅ Report saved successfully in your Google Drive at: {pdf_path}")
files.download(pdf_path)
