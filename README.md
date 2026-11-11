# Toobit Futures Scanner – MTF + TP4 + leverage hint

Streamlit scanner for public Toobit USDT perpetual market data. It does not place orders.

## Run
`pip install -r requirements.txt`
`streamlit run app.py`

Leverage hint is a conservative heuristic capped at 3x, based on stop distance and signal score. It is not an exchange maximum, not a position-size calculation, and not financial advice. Verify the live leverage/risk tier on the Toobit contract page.
