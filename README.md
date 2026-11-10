# Toobit Mobile Futures Scanner

Streamlit dashboard for public Toobit USDT-M market data. It does not connect to an account and does not place orders.

## Multi-timeframe logic
- 15m: establishes directional bias using EMA12/EMA26 and price position.
- 5m: confirms developing momentum and volume.
- 1m: used only as a short-term entry trigger; it is noisy and should not be used alone.
- Signal states: Early Alert, Entry Confirmed, Waiting for Confirmation, and Expired/Late.
- Late signals are penalized when price has travelled more than 1.2 ATR(5m) beyond the recent 5m swing trigger.

Scores are heuristic rankings, not probabilities or guaranteed trade signals. Entry/SL/TP levels are indicative and require manual validation. OI change is based on snapshots stored by the running app and is not a directional indicator by itself.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy on Streamlit Community Cloud using `app.py` as the main file.
