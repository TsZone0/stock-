from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import yfinance as yf
import time
from datetime import datetime
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

NIFTY50 = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","ITC.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
    "BAJFINANCE.NS","WIPRO.NS","HCLTECH.NS","SUNPHARMA.NS","ONGC.NS",
    "NTPC.NS","POWERGRID.NS","ULTRACEMCO.NS","TECHM.NS","NESTLEIND.NS",
    "ADANIENT.NS","JSWSTEEL.NS","TATAMOTORS.NS","M&M.NS","BAJAJFINSV.NS",
    "DIVISLAB.NS","DRREDDY.NS","CIPLA.NS","EICHERMOT.NS","HEROMOTOCO.NS",
    "BRITANNIA.NS","COALINDIA.NS","BPCL.NS","GRASIM.NS","INDUSINDBK.NS",
    "TATACONSUM.NS","APOLLOHOSP.NS","ADANIPORTS.NS","SBILIFE.NS",
    "HDFCLIFE.NS","TATASTEEL.NS","UPL.NS","SHRIRAMFIN.NS","HINDALCO.NS"
]

def safe(val, decimals=2):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), decimals)
    except:
        return None

def fmt_num(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return str(val)
    except:
        return "N/A"

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta).where(delta < 0, 0.0).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-10)
    return round(float(100 - (100 / (1 + rs.iloc[-1]))), 2)

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal= macd.ewm(span=9, adjust=False).mean()
    hist  = macd - signal
    return round(float(macd.iloc[-1]),2), round(float(signal.iloc[-1]),2), round(float(hist.iloc[-1]),2)

def compute_bb(series, period=20):
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + 2*std
    lower = ma - 2*std
    return round(float(upper.iloc[-1]),2), round(float(ma.iloc[-1]),2), round(float(lower.iloc[-1]),2)

def compute_stoch(hist, period=14):
    low_min  = hist['Low'].rolling(period).min()
    high_max = hist['High'].rolling(period).max()
    k = 100 * (hist['Close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(3).mean()
    return round(float(k.iloc[-1]),2), round(float(d.iloc[-1]),2)

def compute_atr(hist, period=14):
    h, l, c = hist['High'], hist['Low'], hist['Close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]),2)

def get_stock_basic(ticker):
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        if hist.empty or len(hist) < 20:
            return None

        symbol     = ticker.replace(".NS", "")
        ltp        = round(float(hist['Close'].iloc[-1]), 2)
        prev_close = round(float(hist['Close'].iloc[-2]), 2)
        open_price = round(float(hist['Open'].iloc[-1]), 2)
        high       = round(float(hist['High'].iloc[-1]), 2)
        low        = round(float(hist['Low'].iloc[-1]), 2)
        change_pct = round(((ltp - prev_close) / prev_close) * 100, 2)

        vol_today = int(hist['Volume'].iloc[-1])
        vol_avg20 = int(hist['Volume'].tail(21).iloc[:-1].mean())
        vol_ratio = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else 0

        rsi   = compute_rsi(hist['Close'])
        ema20 = round(float(hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]), 2)
        ema50 = round(float(hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1]), 2)

        w52_high = round(float(hist['High'].max()), 2)
        w52_low  = round(float(hist['Low'].min()), 2)
        from_52h = round(((ltp - w52_high) / w52_high) * 100, 2)
        from_52l = round(((ltp - w52_low) / w52_low) * 100, 2)

        recent_high = round(float(hist['High'].tail(21).iloc[:-1].max()), 2)
        recent_low  = round(float(hist['Low'].tail(21).iloc[:-1].min()), 2)
        breakout = "NONE"
        if ltp >= recent_high: breakout = "BULLISH"
        elif ltp <= recent_low: breakout = "BEARISH"

        if ltp > ema20 and ema20 > ema50: trend = "UPTREND"
        elif ltp < ema20 and ema20 < ema50: trend = "DOWNTREND"
        else: trend = "SIDEWAYS"

        return {
            "symbol": symbol, "ltp": ltp, "open": open_price,
            "high": high, "low": low, "prev_close": prev_close,
            "change_pct": change_pct, "volume": vol_today,
            "vol_avg20": vol_avg20, "vol_ratio": vol_ratio,
            "rsi": rsi, "ema20": ema20, "ema50": ema50,
            "w52_high": w52_high, "w52_low": w52_low,
            "from_52h": from_52h, "from_52l": from_52l,
            "breakout": breakout, "trend": trend,
        }
    except Exception as e:
        print(f"  Error {ticker}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse')
def analyse():
    return render_template('analyse.html')

@app.route('/api/stocks')
def get_stocks():
    results = []
    for ticker in NIFTY50:
        print(f"Fetching {ticker}...")
        data = get_stock_basic(ticker)
        if data:
            results.append(data)
        time.sleep(0.1)
    results.sort(key=lambda x: x['change_pct'])
    return jsonify({"status":"success","timestamp":datetime.now().strftime("%d %b %Y, %I:%M %p"),"count":len(results),"data":results})

@app.route('/api/analyse/<symbol>')
def analyse_stock(symbol):
    ticker = symbol.upper() + ".NS"
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        info = tk.info

        if hist.empty or len(hist) < 30:
            return jsonify({"status":"error","message":"Stock data nahi mila. Symbol check karo."})

        ltp        = round(float(hist['Close'].iloc[-1]), 2)
        prev_close = round(float(hist['Close'].iloc[-2]), 2)
        change_pct = round(((ltp - prev_close) / prev_close) * 100, 2)
        open_p     = round(float(hist['Open'].iloc[-1]), 2)
        high       = round(float(hist['High'].iloc[-1]), 2)
        low        = round(float(hist['Low'].iloc[-1]), 2)
        vol        = int(hist['Volume'].iloc[-1])
        vol_avg    = int(hist['Volume'].tail(21).iloc[:-1].mean())
        vol_ratio  = round(vol / vol_avg, 2) if vol_avg > 0 else 0

        # Technical indicators
        rsi                  = compute_rsi(hist['Close'])
        macd, macd_sig, macd_hist = compute_macd(hist['Close'])
        bb_up, bb_mid, bb_low     = compute_bb(hist['Close'])
        stoch_k, stoch_d         = compute_stoch(hist)
        atr                      = compute_atr(hist)

        ema9  = round(float(hist['Close'].ewm(span=9,  adjust=False).mean().iloc[-1]),2)
        ema20 = round(float(hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]),2)
        ema50 = round(float(hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1]),2)
        ema200= round(float(hist['Close'].ewm(span=200,adjust=False).mean().iloc[-1]),2)

        sma20 = round(float(hist['Close'].rolling(20).mean().iloc[-1]),2)
        sma50 = round(float(hist['Close'].rolling(50).mean().iloc[-1]),2)

        w52_high = round(float(hist['High'].max()),2)
        w52_low  = round(float(hist['Low'].min()),2)
        from_52h = round(((ltp - w52_high)/w52_high)*100,2)
        from_52l = round(((ltp - w52_low)/w52_low)*100,2)

        recent_high = round(float(hist['High'].tail(21).iloc[:-1].max()),2)
        recent_low  = round(float(hist['Low'].tail(21).iloc[:-1].min()),2)

        if ltp >= recent_high: breakout = "BULLISH"
        elif ltp <= recent_low: breakout = "BEARISH"
        else: breakout = "NONE"

        if ltp > ema20 > ema50: trend = "UPTREND"
        elif ltp < ema20 < ema50: trend = "DOWNTREND"
        else: trend = "SIDEWAYS"

        # Support / Resistance (pivot)
        pivot = round((high + low + ltp)/3, 2)
        r1    = round(2*pivot - low, 2)
        r2    = round(pivot + (high-low), 2)
        s1    = round(2*pivot - high, 2)
        s2    = round(pivot - (high-low), 2)

        # Price history for mini chart (last 60 days)
        chart_dates  = [str(d.date()) for d in hist.index[-60:]]
        chart_close  = [round(float(v),2) for v in hist['Close'].iloc[-60:]]
        chart_volume = [int(v) for v in hist['Volume'].iloc[-60:]]

        # Fundamental data
        def g(key): return fmt_num(info.get(key))

        fundamental = {
            "company_name":   info.get("longName", symbol),
            "sector":         info.get("sector","N/A"),
            "industry":       info.get("industry","N/A"),
            "market_cap":     safe(info.get("marketCap"), 0),
            "pe_ratio":       safe(info.get("trailingPE")),
            "forward_pe":     safe(info.get("forwardPE")),
            "pb_ratio":       safe(info.get("priceToBook")),
            "eps":            safe(info.get("trailingEps")),
            "eps_forward":    safe(info.get("forwardEps")),
            "dividend_yield": safe(info.get("dividendYield",0) * 100 if info.get("dividendYield") else 0),
            "book_value":     safe(info.get("bookValue")),
            "roe":            safe(info.get("returnOnEquity",0)*100 if info.get("returnOnEquity") else 0),
            "roa":            safe(info.get("returnOnAssets",0)*100 if info.get("returnOnAssets") else 0),
            "debt_to_equity": safe(info.get("debtToEquity")),
            "current_ratio":  safe(info.get("currentRatio")),
            "revenue":        safe(info.get("totalRevenue"),0),
            "gross_margin":   safe(info.get("grossMargins",0)*100 if info.get("grossMargins") else 0),
            "operating_margin": safe(info.get("operatingMargins",0)*100 if info.get("operatingMargins") else 0),
            "profit_margin":  safe(info.get("profitMargins",0)*100 if info.get("profitMargins") else 0),
            "free_cashflow":  safe(info.get("freeCashflow"),0),
            "description":    info.get("longBusinessSummary","N/A")[:500] if info.get("longBusinessSummary") else "N/A",
            "employees":      info.get("fullTimeEmployees","N/A"),
            "website":        info.get("website","N/A"),
            "beta":           safe(info.get("beta")),
            "avg_volume":     safe(info.get("averageVolume"),0),
        }

        technical = {
            "ltp": ltp, "open": open_p, "high": high, "low": low,
            "prev_close": prev_close, "change_pct": change_pct,
            "volume": vol, "vol_avg": vol_avg, "vol_ratio": vol_ratio,
            "rsi": rsi,
            "macd": macd, "macd_signal": macd_sig, "macd_hist": macd_hist,
            "bb_upper": bb_up, "bb_mid": bb_mid, "bb_lower": bb_low,
            "stoch_k": stoch_k, "stoch_d": stoch_d,
            "atr": atr,
            "ema9": ema9, "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "sma20": sma20, "sma50": sma50,
            "w52_high": w52_high, "w52_low": w52_low,
            "from_52h": from_52h, "from_52l": from_52l,
            "breakout": breakout, "trend": trend,
            "pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2,
        }

        return jsonify({
            "status": "success",
            "symbol": symbol.upper(),
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "fundamental": fundamental,
            "technical": technical,
            "chart": {"dates": chart_dates, "close": chart_close, "volume": chart_volume}
        })

    except Exception as e:
        return jsonify({"status":"error","message":str(e)})

if __name__ == '__main__':
    print("🚀 Server: http://localhost:5000")
    app.run(debug=True, port=5000)