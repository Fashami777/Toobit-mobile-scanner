import json, time, math
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Toobit public USDT-M market-data endpoints. No API key and no order placement.
BASE = "https://api.toobit.com"
STATE_FILE = Path("scanner_state.json")
TIMEOUT = 12

st.set_page_config(page_title="Toobit Pump & Dump Scanner", layout="wide")
st.title("Toobit Futures | Pump & Dump Scanner")
st.caption("اسکنر پژوهشی بازار USDT-M؛ تحلیل ۱۵m/۵m/۱m، فقط داده عمومی و بدون ثبت سفارش.")

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;}
h1 {font-size: clamp(1.35rem, 5vw, 2.1rem) !important; line-height: 1.25 !important;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: .65rem .75rem; border-radius: 12px;}
[data-testid="stMetricValue"] {font-size: clamp(1.1rem, 4vw, 1.7rem) !important;}
div[data-testid="stTabs"] button {font-size: .9rem; padding: .45rem .55rem;}
.mobile-card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 12px 14px; margin: 9px 0; background: rgba(128,128,128,.045);}
.mobile-head {display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;}
.mobile-symbol {font-size:1.05rem; font-weight:750;}
.mobile-score {font-size:1.15rem; font-weight:800;}
.mobile-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; margin-top:10px;}
.mobile-cell {padding:7px 9px; border-radius:9px; background:rgba(128,128,128,.09); min-width:0;}
.mobile-label {font-size:.72rem; opacity:.72;}
.mobile-value {font-size:.92rem; font-weight:650; overflow-wrap:anywhere;}
.mobile-reasons {font-size:.82rem; margin-top:9px; line-height:1.6; opacity:.9;}
div[data-testid="stDataFrame"] {font-size:.8rem;}
@media (max-width: 600px) {
  .block-container {padding-left:.75rem; padding-right:.75rem;}
  [data-testid="stHorizontalBlock"] {gap:.45rem;}
  [data-testid="stMetric"] {padding:.5rem .55rem;}
  [data-testid="stMetricLabel"] {font-size:.72rem !important;}
  [data-testid="stMetricValue"] {font-size:1.05rem !important;}
  div[data-testid="stTabs"] button {font-size:.78rem; padding:.35rem .35rem;}
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("تنظیمات اسکن")
    refresh_sec = st.selectbox("فاصله به‌روزرسانی", [300, 600, 900], index=0, format_func=lambda x: f"{x//60} دقیقه")
    top_universe = st.slider("تعداد ارزهای پُرحجم برای تحلیل کندل", 15, 100, 40, 5)
    oi_count = st.slider("تعداد ارز برای بررسی OI و دفتر سفارش", 5, 40, 20, 5)
    min_quote_volume = st.number_input("حداقل گردش ۲۴ ساعته (USDT)", min_value=0, value=100000, step=50000)
    min_score = st.slider("حداقل امتیاز نمایش", 0, 100, 50, 5)
    st.info("برای شروع، تنظیمات پیش‌فرض را نگه دار. درخواست‌های API با مکث و محدودیت انجام می‌شوند.")

# Auto-refresh counter: Streamlit reruns the script at the selected interval.
refresh_count = st_autorefresh(interval=refresh_sec * 1000, key="scanner_refresh_v2", limit=None)

# Manual refresh is useful for diagnosing whether reruns are working.
with st.sidebar:
    st.caption(f"چرخه اجرا: {refresh_count}")
    if st.button("اسکن مجدد همین حالا", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

session = requests.Session()
session.headers.update({"User-Agent": "ToobitResearchScanner/1.0"})

def get(path, params=None):
    r = session.get(BASE + path, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def as_float(x, default=np.nan):
    try: return float(x)
    except (TypeError, ValueError): return default

def rows_from_payload(x, key=None):
    if isinstance(x, list): return x
    if isinstance(x, dict):
        if key and isinstance(x.get(key), list): return x[key]
        if isinstance(x.get("data"), list): return x["data"]
    return []

def load_state():
    try: return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception: return {}

def save_state(state):
    try: STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except Exception: pass

def candle_frame(payload):
    # Toobit kline schema: [openTime, open, high, low, close, volume, closeTime, quoteVolume, trades, takerBuyBase, takerBuyQuote]
    if not isinstance(payload, list) or not payload: return None
    cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote"]
    df = pd.DataFrame(payload)
    if df.shape[1] < 6: return None
    df = df.iloc[:, :min(len(cols), df.shape[1])]
    df.columns = cols[:df.shape[1]]
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close","volume"]).reset_index(drop=True)

def rsi(s, period=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def analyze_candles(df):
    if df is None or len(df) < 30: return None
    # Exclude currently-forming candle for indicators/signals.
    d = df.iloc[:-1].copy()
    if len(d) < 25: return None
    close, vol = d.close, d.volume
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    rv = vol.iloc[-1] / max(vol.iloc[-21:-1].mean(), 1e-12)
    r = float(rsi(close).iloc[-1])
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std(ddof=0)
    upper, lower = mid + 2*sd, mid - 2*sd
    width = float(((upper-lower)/mid).iloc[-1]) if mid.iloc[-1] else np.nan
    width_avg = float(((upper-lower)/mid).iloc[-20:-1].mean())
    squeeze = bool(width < width_avg * 0.80) if np.isfinite(width_avg) else False
    resistance = float(d.high.iloc[-21:-1].max())
    support = float(d.low.iloc[-21:-1].min())
    price = float(close.iloc[-1])
    breakout_up = price > resistance
    breakout_dn = price < support
    macd_up = float(macd.iloc[-1]) > float(signal.iloc[-1]) and float(macd.iloc[-1]-signal.iloc[-1]) > float(macd.iloc[-2]-signal.iloc[-2])
    macd_dn = float(macd.iloc[-1]) < float(signal.iloc[-1]) and float(macd.iloc[-1]-signal.iloc[-1]) < float(macd.iloc[-2]-signal.iloc[-2])
    # Candle returns and true range estimate for illustrative risk levels.
    ret15 = (price / float(d.close.iloc[-2]) - 1) * 100
    tr = pd.concat([(d.high-d.low), (d.high-d.close.shift()).abs(), (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    ema12_now, ema26_now = float(ema12.iloc[-1]), float(ema26.iloc[-1])
    return dict(price=price, rsi=r, volume_ratio=float(rv), squeeze=squeeze,
                trend_up=ema12_now > ema26_now and price > ema26_now,
                trend_dn=ema12_now < ema26_now and price < ema26_now,
                breakout_up=breakout_up, breakout_dn=breakout_dn, macd_up=macd_up, macd_dn=macd_dn,
                resistance=resistance, support=support, atr=atr, ret15=ret15,
                candle_time=int(d.open_time.iloc[-1]))


def swing_levels(df, kind, lookback=120):
    """Confirmed local swing highs/lows from completed candles only."""
    if df is None or len(df) < 12:
        return []
    d = df.iloc[:-1].tail(lookback).reset_index(drop=True)
    col = "high" if kind == "resistance" else "low"
    vals = d[col].astype(float).to_numpy()
    found = []
    # A pivot is confirmed only after two candles have formed on its right.
    for i in range(2, len(vals)-2):
        w = vals[i-2:i+3]
        if kind == "resistance" and vals[i] >= np.max(w) and np.count_nonzero(w == vals[i]) == 1:
            found.append(float(vals[i]))
        elif kind == "support" and vals[i] <= np.min(w) and np.count_nonzero(w == vals[i]) == 1:
            found.append(float(vals[i]))
    return found

def merge_levels(levels, tolerance=0.0025):
    """Merge nearby swing levels to avoid treating one zone as many targets."""
    out = []
    for level in sorted(x for x in levels if np.isfinite(x) and x > 0):
        if not out or abs(level-out[-1])/out[-1] > tolerance:
            out.append(level)
        else:
            out[-1] = (out[-1] + level) / 2
    return out

def structural_targets(entry, direction, frames, atr):
    """Return four nearest confirmed structural levels, nearest first."""
    kind = "resistance" if direction == "LONG" else "support"
    levels = merge_levels([v for frame in frames for v in swing_levels(frame, kind)])
    if direction == "LONG":
        targets = [v for v in levels if v > entry * 1.001]
        targets.sort()
    else:
        targets = [v for v in levels if v < entry * 0.999]
        targets.sort(reverse=True)
    return targets[:4]

def structural_stop(entry, direction, frames, atr):
    kind = "support" if direction == "LONG" else "resistance"
    levels = merge_levels([v for frame in frames for v in swing_levels(frame, kind)])
    if direction == "LONG":
        candidates = [v for v in levels if v < entry]
        structural = max(candidates) if candidates else np.nan
        fallback = entry - 1.2 * atr
        return structural if np.isfinite(structural) and (entry-structural) <= 3*atr else fallback
    candidates = [v for v in levels if v > entry]
    structural = min(candidates) if candidates else np.nan
    fallback = entry + 1.2 * atr
    return structural if np.isfinite(structural) and (structural-entry) <= 3*atr else fallback

def level_metrics(entry, stop, targets, direction):
    risk = abs(entry-stop)
    result = {}
    for i in range(4):
        target = targets[i] if i < len(targets) else np.nan
        reward = abs(target-entry) if np.isfinite(target) else np.nan
        rr = reward/risk if risk > 0 and np.isfinite(reward) else np.nan
        pct = ((target/entry)-1)*100 if direction == "LONG" and np.isfinite(target) else (((entry-target)/entry)*100 if direction == "SHORT" and np.isfinite(target) else np.nan)
        result[f"{direction} TP{i+1}*"] = target
        result[f"{direction} TP{i+1} %"] = pct
        result[f"{direction} TP{i+1} R/R"] = rr
    return result

def score_row(a, funding, oi_change, book_ratio):
    # Transparent heuristic score, not a probability.
    long = short = 0
    long_reasons, short_reasons = [], []
    if a["volume_ratio"] >= 2:
        long += 15; short += 15
        long_reasons.append("جهش حجم"); short_reasons.append("جهش حجم")
    if oi_change is not None and np.isfinite(oi_change) and oi_change >= 1.5:
        long += 15; short += 15
        long_reasons.append("افزایش OI"); short_reasons.append("افزایش OI")
    if a["breakout_up"]:
        long += 20; long_reasons.append("شکست مقاومت")
    if a["breakout_dn"]:
        short += 20; short_reasons.append("شکست حمایت")
    if a["macd_up"]:
        long += 10; long_reasons.append("شتاب MACD صعودی")
    if a["macd_dn"]:
        short += 10; short_reasons.append("شتاب MACD نزولی")
    if 50 <= a["rsi"] <= 70:
        long += 10; long_reasons.append("RSI سازگار با مومنتوم صعودی")
    if 30 <= a["rsi"] <= 50:
        short += 10; short_reasons.append("RSI سازگار با مومنتوم نزولی")
    if a["squeeze"]:
        long += 10; short += 10
        long_reasons.append("فشردگی بولینگر"); short_reasons.append("فشردگی بولینگر")
    if book_ratio is not None:
        if book_ratio >= 1.20:
            long += 10; long_reasons.append("غلبه عمق خرید")
        elif book_ratio <= 1/1.20:
            short += 10; short_reasons.append("غلبه عمق فروش")
    # Funding is only a contrarian crowding hint; modest weight and no standalone direction.
    if funding is not None and np.isfinite(funding):
        if funding < -0.0002:
            long += 10; long_reasons.append("Funding منفی")
        elif funding > 0.0002:
            short += 10; short_reasons.append("Funding مثبت")
    return min(long,100), min(short,100), long_reasons, short_reasons

def combine_timeframes(a15, a5, a1):
    """15m establishes bias, 5m confirms setup, 1m is only an entry trigger."""
    long_score = short_score = 0
    long_reasons, short_reasons = [], []
    if a15["trend_up"]:
        long_score += 25; long_reasons.append("روند ۱۵m صعودی")
    if a15["trend_dn"]:
        short_score += 25; short_reasons.append("روند ۱۵m نزولی")
    if a5["trend_up"] and (a5["macd_up"] or a5["breakout_up"]):
        long_score += 25; long_reasons.append("تأیید حرکت در ۵m")
    if a5["trend_dn"] and (a5["macd_dn"] or a5["breakout_dn"]):
        short_score += 25; short_reasons.append("تأیید حرکت در ۵m")
    if a1["macd_up"] or a1["breakout_up"]:
        long_score += 15; long_reasons.append("تریگر ورود ۱m")
    if a1["macd_dn"] or a1["breakout_dn"]:
        short_score += 15; short_reasons.append("تریگر ورود ۱m")
    # Volume confirmation is taken from 5m, not from a still-forming candle.
    if a5["volume_ratio"] >= 1.5:
        long_score += 10; short_score += 10
        long_reasons.append("افزایش حجم ۵m"); short_reasons.append("افزایش حجم ۵m")
    # Early means bias + developing 5m setup, before 1m entry trigger.
    long_early = a15["trend_up"] and a5["trend_up"] and not (a1["macd_up"] or a1["breakout_up"])
    short_early = a15["trend_dn"] and a5["trend_dn"] and not (a1["macd_dn"] or a1["breakout_dn"])
    long_confirmed = a15["trend_up"] and a5["trend_up"] and (a1["macd_up"] or a1["breakout_up"])
    short_confirmed = a15["trend_dn"] and a5["trend_dn"] and (a1["macd_dn"] or a1["breakout_dn"])
    # Avoid chasing: if price has already travelled >1.2 ATR(5m) beyond its trigger, expire it.
    atr5 = a5["atr"] if np.isfinite(a5["atr"]) and a5["atr"] > 0 else a5["price"] * 0.005
    long_extension = max(0.0, a1["price"] - a5["resistance"]) / atr5
    short_extension = max(0.0, a5["support"] - a1["price"]) / atr5
    long_expired = long_extension > 1.2
    short_expired = short_extension > 1.2
    if long_expired: long_score = max(0, long_score - 35); long_reasons.append("حرکت از نقطه شکست فاصله گرفته؛ ورود تعقیبی ممنوع")
    if short_expired: short_score = max(0, short_score - 35); short_reasons.append("حرکت از نقطه شکست فاصله گرفته؛ ورود تعقیبی ممنوع")
    long_status = "منقضی / دیرهنگام" if long_expired else ("ورود تأییدشده" if long_confirmed else ("هشدار زودهنگام" if long_early else "در انتظار تأیید"))
    short_status = "منقضی / دیرهنگام" if short_expired else ("ورود تأییدشده" if short_confirmed else ("هشدار زودهنگام" if short_early else "در انتظار تأیید"))
    return (min(long_score,100), min(short_score,100), long_status, short_status,
            "، ".join(long_reasons), "، ".join(short_reasons), a1["price"], atr5)

def get_oi(symbol):
    try:
        x = get("/quote/v1/openInterest", {"symbol": symbol})
        arr = rows_from_payload(x, "openInterestList")
        return as_float(arr[0].get("size")) if arr else np.nan
    except Exception: return np.nan

def get_book_ratio(symbol):
    try:
        x = get("/quote/v1/depth", {"symbol": symbol, "limit": 20})
        bids, asks = x.get("b", []), x.get("a", [])
        bv = sum(as_float(z[0],0)*as_float(z[1],0) for z in bids)
        av = sum(as_float(z[0],0)*as_float(z[1],0) for z in asks)
        return bv/av if av > 0 else np.nan
    except Exception: return np.nan

@st.cache_data(ttl=240, show_spinner=False)
def fetch_market(min_quote_volume, top_universe):
    info = get("/api/v1/exchangeInfo")
    contracts = [c for c in info.get("contracts", []) if c.get("status") == "TRADING"
                 and c.get("inverse") is False and c.get("symbol","").endswith("-SWAP-USDT")]
    tickers = rows_from_payload(get("/quote/v1/contract/ticker/24hr"))
    funds = rows_from_payload(get("/api/v1/futures/fundingRate"))
    tmap = {x.get("s"): x for x in tickers}
    fmap = {x.get("symbol"): as_float(x.get("rate")) for x in funds}
    valid = []
    for c in contracts:
        sym = c["symbol"]
        t = tmap.get(sym)
        if not t: continue
        qv = as_float(t.get("qv"), 0)
        if qv >= min_quote_volume:
            valid.append((sym, qv, t, fmap.get(sym, np.nan)))
    valid.sort(key=lambda x: x[1], reverse=True)
    return valid[:top_universe]


def render_mobile_cards(frame, direction):
    score_col = "Long" if direction == "LONG" else "Short"
    trigger_col = "Long trigger" if direction == "LONG" else "Short trigger"
    sl_col = f"{direction} SL*"
    reason_col = "Long reasons" if direction == "LONG" else "Short reasons"
    frame = frame.sort_values(score_col, ascending=False).head(10)
    if frame.empty:
        st.info("در این چرخه نامزدی با این حداقل امتیاز پیدا نشد.")
        return
    for _, row in frame.iterrows():
        status_col = "Signal status LONG" if direction == "LONG" else "Signal status SHORT"
        distance_col = "Long distance %" if direction == "LONG" else "Short distance %"
        cells = [("وضعیت سیگنال", str(row[status_col])),
                 ("قیمت فعلی / مبنای محاسبه", f"{row['Price']:.8g}"),
                 ("محرک قیمت", f"{row[trigger_col]:.8g}"),
                 ("فاصله از محرک", f"{row[distance_col]:+.2f}%"),
                 ("حد ضرر", f"{row[sl_col]:.8g}  ({row[f'{direction} SL %']:.2f}% زیان)"),
                 ("اهرم پیشنهادی", str(row[f'{direction} Leverage']))]
        for n in range(1, 5):
            target = row[f"{direction} TP{n}*"]
            pct = row[f"{direction} TP{n} %"]
            rr = row[f"{direction} TP{n} R/R"]
            label = f"{target:.8g} ({pct:+.2f}%) · R/R {rr:.2f}" if np.isfinite(target) else "سطح معتبر پیدا نشد"
            cells.append((f"هدف TP{n}", label))
        cells.extend([("تغییر ۱۵ دقیقه", f"{row['15m %']:+.2f}%"),
                      ("RSI ۱m / ۵m", f"{row['1m RSI']:.1f} / {row['5m RSI']:.1f}"),
                      ("جهش حجم", f"{row['Vol x']:.2f}×"),
                      ("تغییر OI", f"{row['OI Δ %']:.2f}%" if np.isfinite(row['OI Δ %']) else "نامشخص")])
        grid = "".join(f'<div class="mobile-cell"><div class="mobile-label">{label}</div><div class="mobile-value">{value}</div></div>' for label, value in cells)
        st.markdown(f"""<div class="mobile-card"><div class="mobile-head"><span class="mobile-symbol">{row['Symbol']}</span><span class="mobile-score">{direction} · {int(row[score_col])}/100</span></div><div class="mobile-grid">{grid}</div><div class="mobile-reasons">دلایل: {row[reason_col] or 'هنوز تأیید کافی ثبت نشده'}</div></div>""", unsafe_allow_html=True)

def suggest_leverage(stop_pct, score, status):
    """Conservative indicative leverage cap; not a position-size recommendation."""
    if "منقضی" in str(status) or score < 50 or not np.isfinite(stop_pct) or stop_pct <= 0:
        return "عدم ورود / 1x حداکثر", 1
    # Wider stop / higher volatility => lower leverage. Hard cap deliberately kept low.
    if stop_pct >= 2.5:
        lev = 1
    elif stop_pct >= 1.5:
        lev = 2
    else:
        lev = 3
    if score < 70:
        lev = min(lev, 1)
    return f"{lev}x (محافظه‌کارانه)", lev


def safe_signal(score):
    if score >= 85: return "Very Strong"
    if score >= 70: return "Strong"
    if score >= 50: return "Watch"
    return "ضعیف"

try:
    with st.spinner("دریافت فهرست قراردادها و داده‌های بازار از Toobit..."):
        universe = fetch_market(min_quote_volume, top_universe)
    if not universe:
        st.warning("در حال حاضر داده‌ای دریافت نشد. اتصال اینترنت، دسترسی منطقه‌ای و API را بررسی کن.")
        st.stop()

    state = load_state()
    rows = []
    scan_errors = []
    progress = st.progress(0)
    status = st.empty()
    # Analyze candles for liquid universe; expensive OI/depth only for the highest-volume subset.
    for i, (sym, qv, ticker, funding) in enumerate(universe):
        status.caption(f"در حال تحلیل {sym} ({i+1}/{len(universe)})")
        try:
            raw15 = get("/quote/v1/klines", {"symbol": sym, "interval": "15m", "limit": 80})
            raw5 = get("/quote/v1/klines", {"symbol": sym, "interval": "5m", "limit": 80})
            raw1 = get("/quote/v1/klines", {"symbol": sym, "interval": "1m", "limit": 80})
            raw1h = get("/quote/v1/klines", {"symbol": sym, "interval": "1h", "limit": 120})
            df15, df5, df1, df1h = (candle_frame(z) for z in (raw15, raw5, raw1, raw1h))
            a, a5, a1 = (analyze_candles(z) for z in (df15, df5, df1))
            if not a or not a5 or not a1: continue
            oi_delta, book = np.nan, np.nan
            if i < oi_count:
                oi = get_oi(sym)
                old = state.get(sym, {}).get("oi")
                if np.isfinite(oi) and old and float(old) > 0:
                    oi_delta = (oi/float(old)-1)*100
                if np.isfinite(oi):
                    state[sym] = {"oi": oi, "time": time.time()}
                book = get_book_ratio(sym)
                time.sleep(0.12)
            base_l, base_s, base_lr, base_sr = score_row(a, funding, oi_delta, book)
            mtf_l, mtf_s, lstatus, sstatus, mlr, msr, entry_price, atr5 = combine_timeframes(a, a5, a1)
            # Blend legacy market context with multi-timeframe confirmation; not a probability.
            ls = min(100, int(0.45 * base_l + 0.55 * mtf_l))
            ss = min(100, int(0.45 * base_s + 0.55 * mtf_s))
            lr = base_lr + ([mlr] if mlr else [])
            sr = base_sr + ([msr] if msr else [])
            # Structural targets from confirmed swing levels on 5m/15m/1h.
            atr = a5["atr"] if np.isfinite(a5["atr"]) and a5["atr"] > 0 else a1["price"]*0.005
            entry_price = float(entry_price)
            frames_for_levels = [df5, df15, df1h]
            long_stop = structural_stop(entry_price, "LONG", frames_for_levels, atr)
            short_stop = structural_stop(entry_price, "SHORT", frames_for_levels, atr)
            long_targets = structural_targets(entry_price, "LONG", frames_for_levels, atr)
            short_targets = structural_targets(entry_price, "SHORT", frames_for_levels, atr)
            long_levels = level_metrics(entry_price, long_stop, long_targets, "LONG")
            short_levels = level_metrics(entry_price, short_stop, short_targets, "SHORT")
            long_risk_pct = abs(entry_price-long_stop)/entry_price*100 if entry_price > 0 else np.nan
            short_risk_pct = abs(short_stop-entry_price)/entry_price*100 if entry_price > 0 else np.nan
            long_rr1 = long_levels["LONG TP1 R/R"]
            short_rr1 = short_levels["SHORT TP1 R/R"]
            long_room_ok = np.isfinite(long_rr1) and long_rr1 >= 1.5
            short_room_ok = np.isfinite(short_rr1) and short_rr1 >= 1.5
            if not long_room_ok:
                ls = min(ls, 45); lr.append("فضای کافی تا مقاومت بعدی وجود ندارد یا R/R کمتر از 1.5 است")
                lstatus = "عدم ورود؛ هدف ساختاری نامعتبر"
            if not short_room_ok:
                ss = min(ss, 45); sr.append("فضای کافی تا حمایت بعدی وجود ندارد یا R/R کمتر از 1.5 است")
                sstatus = "عدم ورود؛ هدف ساختاری نامعتبر"
            rows.append({
                "Symbol": sym.replace("-SWAP-USDT","/USDT"),
                "Price": a1["price"], "15m %": a["ret15"],
                "Signal status LONG": lstatus, "Signal status SHORT": sstatus,
                "Long distance %": ((a1["price"] / a5["resistance"] - 1) * 100) if a5["resistance"] else np.nan,
                "Short distance %": ((a1["price"] / a5["support"] - 1) * 100) if a5["support"] else np.nan,
                "1m RSI": a1["rsi"], "5m RSI": a5["rsi"], "Vol x": a["volume_ratio"],
                "RSI": a["rsi"], "OI Δ %": oi_delta, "Funding %": funding*100 if np.isfinite(funding) else np.nan,
                "Bid/Ask depth": book, "Long": ls, "Short": ss,
                "Long reasons": "، ".join(lr), "Short reasons": "، ".join(sr),
                "Long trigger": a5["resistance"], "Short trigger": a5["support"],
                # Reference entry is the latest completed 1m close; risk distance uses 5m ATR.
                "LONG SL*": long_stop, "LONG SL %": long_risk_pct,
                "SHORT SL*": short_stop, "SHORT SL %": short_risk_pct,
                "LONG Leverage": suggest_leverage(long_risk_pct, ls, lstatus)[0],
                "SHORT Leverage": suggest_leverage(short_risk_pct, ss, sstatus)[0],
                **long_levels, **short_levels,
                "LONG Target valid": long_room_ok, "SHORT Target valid": short_room_ok,
                "Signal candle UTC": datetime.fromtimestamp(a["candle_time"]/1000, timezone.utc).strftime("%H:%M")
            })
        except Exception as e:
            scan_errors.append(f"{sym}: {type(e).__name__}: {e}")
            continue
        progress.progress((i+1)/len(universe))
    save_state(state)
    status.empty(); progress.empty()

    if not rows:
        st.error("کندل قابل تحلیل دریافت نشد. احتمالاً قالب پاسخ API تغییر کرده یا دسترسی API محدود است.")
        st.stop()
    df = pd.DataFrame(rows)
    now_utc = datetime.now(timezone.utc)
    st.caption(f"آخرین بررسی موفق: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} | چرخه اجرا: {refresh_count} | ارزهای بررسی‌شده: {len(df)} | فاصله اسکن: {refresh_sec//60} دقیقه")
    if scan_errors:
        with st.expander(f"خطاهای دریافت داده ({len(scan_errors)})"):
            st.code("\n".join(scan_errors[:30]))
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ارزهای تحلیل‌شده", len(df))
    c2.metric("نامزدهای LONG ≥ 70", int((df.Long>=70).sum()))
    c3.metric("نامزدهای SHORT ≥ 70", int((df.Short>=70).sum()))
    c4.metric("حجم ۲۴ساعته حداقل", f"{min_quote_volume:,.0f} USDT")

    tab1, tab2, tab3 = st.tabs(["فرصت‌های LONG", "فرصت‌های SHORT", "همه داده‌ها"])
    cols = ["Symbol","Price","15m %","1m RSI","5m RSI","Signal status LONG","Long distance %","Vol x","OI Δ %","Long","Long trigger","LONG SL*","LONG SL %","LONG Leverage","LONG TP1*","LONG TP1 %","LONG TP2*","LONG TP2 %","LONG TP3*","LONG TP3 %","LONG TP4*","LONG TP4 %","LONG TP1 R/R","Long reasons"]
    with tab1:
        x = df[(df.Long >= min_score) & (df["LONG Target valid"])].sort_values("Long", ascending=False)
        render_mobile_cards(x, "LONG")
        with st.expander("نمایش جدول کامل"):
            st.dataframe(x[cols].head(15), use_container_width=True, hide_index=True)
        if x.empty: st.info("در این چرخه نامزدی با این حداقل امتیاز پیدا نشد.")
    cols2 = ["Symbol","Price","15m %","1m RSI","5m RSI","Signal status SHORT","Short distance %","Vol x","OI Δ %","Short","Short trigger","SHORT SL*","SHORT SL %","SHORT Leverage","SHORT TP1*","SHORT TP1 %","SHORT TP2*","SHORT TP2 %","SHORT TP3*","SHORT TP3 %","SHORT TP4*","SHORT TP4 %","SHORT TP1 R/R","Short reasons"]
    with tab2:
        x = df[(df.Short >= min_score) & (df["SHORT Target valid"])].sort_values("Short", ascending=False)
        render_mobile_cards(x, "SHORT")
        with st.expander("نمایش جدول کامل"):
            st.dataframe(x[cols2].head(15), use_container_width=True, hide_index=True)
        if x.empty: st.info("در این چرخه نامزدی با این حداقل امتیاز پیدا نشد.")
    with tab3:
        with st.expander("نمایش جدول کامل همه ارزها"):
            st.dataframe(df.sort_values(["Long","Short"], ascending=False), use_container_width=True, hide_index=True)
        st.download_button("دانلود CSV", df.to_csv(index=False).encode("utf-8-sig"), "toobit_scanner.csv", "text/csv", use_container_width=True)

    st.warning("امتیازها احتمال آماری یا سیگنال قطعی نیستند. OI فقط نسبت به snapshot قبلی همین برنامه مقایسه می‌شود؛ در اجرای اول خالی است و در صورت بستن برنامه/پاک شدن فایل state سابقه از بین می‌رود. نسبت Bid/Ask فقط عمق ۲۰ سطح دفتر سفارش است و سفارش‌های قابل لغو می‌توانند آن را گمراه کنند.")
    st.markdown("**راهنمای سطوح:** Watch = زیر نظر گرفتن | Strong = بررسی دستی پس از تأیید کندل | Very Strong = اولویت بررسی، نه دستور ورود.")
    st.caption("اهرم پیشنهادی صرفاً سقف محافظه‌کارانه و بر اساس فاصله حد ضرر/نوسان و امتیاز سیگنال است (حداکثر 3x)؛ در سیگنال منقضی یا امتیاز زیر 50 ورود پیشنهاد نمی‌شود. اهرم واقعی مجاز هر قرارداد و سطح ریسک آن را در صفحه همان قرارداد Toobit بررسی کن. اهرم به‌تنهایی اندازه پوزیشن را تعیین نمی‌کند؛ اندازه پوزیشن باید با موجودی حساب و درصد ریسک مجاز محاسبه شود. این پیشنهاد تضمین‌کننده جلوگیری از لیکوییدشدن نیست.")
    st.caption("*قیمت مبنای ورود، آخرین کلوز کندل تکمیل‌شده ۱m است. اهداف TP از پیوت‌های سقف/کف تأییدشده در تایم‌فریم‌های ۵m، ۱۵m و ۱h استخراج می‌شوند؛ TP1 نزدیک‌ترین مقاومت/حمایت و TPهای بعدی سطوح دورتر هستند. اگر سطح ساختاری کافی یا نسبت سود به زیان حداقل 1.5 تا TP1 وجود نداشته باشد، آن جهت با وضعیت عدم ورود علامت‌گذاری می‌شود. حد ضرر از نزدیک‌ترین حمایت/مقاومت ساختاری معتبر و در نبود آن از ATR محاسبه می‌شود. این سطوح تخمینی‌اند؛ قبل از ورود دوباره بررسی و کارمزد/اسلیپیج را لحاظ کن.")
except requests.RequestException as e:
    st.error(f"خطای اتصال به API: {e}")
    st.info("اگر API از شبکه شما در دسترس نیست، با VPN یا شبکه دیگری امتحان کن. آدرس پایه API در ابتدای app.py قابل تغییر است.")
except Exception as e:
    st.error(f"خطای اجرای اسکنر: {type(e).__name__}: {e}")
    st.exception(e)
