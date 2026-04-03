import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide
from alpaca.trading.enums import TimeInForce

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Settings
API_KEY    = "API_KEY"
API_SECRET = "API_SECRET_KEY"

SYMBOL         = "NVDA"
BUY_TRESHOLD   =  0.005   # +0.5% → COMPRA
SELL_THRESHOLD = -0.005   # -0.5% → VENDE

# Conexión con el cliente
data_client    = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Llamada a la API para obtener los datos
def fetch_data(symbol: str) -> pd.DataFrame:
    print(f"Descargando datos historicos de {symbol}...")
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 3)  # tres años de datos

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="iex",
    )

    bars = data_client.get_stock_bars(request)
    df   = bars.df.reset_index()

    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    print(f"Descargados {len(df)} datos historicos de {symbol}")
    return df

# Caracteristicas
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"]     = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(5).std()
    df["ma_5"]       = df["close"].rolling(5).mean()
    df["ma_10"]      = df["close"].rolling(10).mean()
    df["ma_20"]      = df["close"].rolling(20).mean()
    df["target"]     = df["close"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df

# Entrenar
def train_model(df: pd.DataFrame):
    print("Entrenando el modelo...")
    features = ["close", "return", "volatility", "ma_5", "ma_10", "ma_20"]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    print(f"    MAE del modelo: ${mae:.2f}")
    return model, features

# Indicar la señal
def get_signal(model, features: list, df: pd.DataFrame) -> tuple:
    latest          = df[features].iloc[-1].values.reshape(1, -1)
    current_price   = float(df["close"].iloc[-1])
    predicted_price = float(model.predict(latest)[0])
    pct_change      = (predicted_price - current_price) / current_price

    if pct_change >= BUY_TRESHOLD:
        signal = "BUY"
    elif pct_change <= SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, current_price, predicted_price, pct_change

# Verificar la posicion
def get_current_position(symbol: str) -> float:
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except Exception:
        return 0.0

# Ejecucion de las ordenes
def execute_order(signal: str, symbol: str, current_price: float):
    position = get_current_position(symbol)
    account  = trading_client.get_account()
    cash     = float(account.cash)

    if signal == "BUY":
        qty = int(cash // current_price)  # compra todo lo que puede con el cash disponible
        if qty == 0:
            print("No hay suficiente cash para comprar ni 1 accion")
            return
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        result = trading_client.submit_order(order)
        print(f"Orden COMPRA ejecutada | {qty} acciones de {symbol} (~${qty * current_price:,.2f})")
        print(f"ID: {result.id}")

    elif signal == "SELL":
        if position > 0:
            qty_to_sell = int(position)  # vende TODO lo que tiene
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty_to_sell,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            result = trading_client.submit_order(order)
            print(f"Orden VENTA ejecutada | {qty_to_sell} acciones de {symbol}")
            print(f"ID: {result.id}")
        else:
            print(f"Señal SELL pero no tienes posicion en {symbol} — sin orden")

    else:
        print(f"Señal HOLD — sin orden ejecutada")

# Status de la cuenta
def print_account_status():
    account = trading_client.get_account()
    print(f"\n  Estado de la cuenta (Paper Trading)")
    print(f"  Cash disponible:   ${float(account.cash):,.2f}")
    print(f"  Valor portafolio:  ${float(account.portfolio_value):,.2f}")
    print(f"  Poder adquisitivo: ${float(account.buying_power):,.2f}\n")

# Principal
def main():
    print(f"\n  NVDA Auto Trader - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Threshold compra: +{BUY_TRESHOLD*100:.1f}%  |  Threshold venta: {SELL_THRESHOLD*100:.1f}%\n")

    # Estado inicial
    print_account_status()

    # Pipeline del modelo
    df_raw  = fetch_data(SYMBOL)
    df_feat = build_features(df_raw)
    model, features = train_model(df_feat)

    # Señal
    print("\nGenerando la señal...")
    signal, current_price, predicted_price, pct_change = get_signal(model, features, df_feat)

    print(f"\n  Señal Generada:  {signal}")
    print(f"  Precio Actual:   ${current_price:.2f}")
    print(f"  Precio Predicho: ${predicted_price:.2f}")
    print(f"  Cambio Esperado: {pct_change * 100:.2f}%")
    print(f"  Posicion Actual: {get_current_position(SYMBOL):.0f} acciones\n")

    # Ejecutar orden
    print("Ejecutando la orden...")
    execute_order(signal, SYMBOL, current_price)

    # Estado final
    print_account_status()

if __name__ == "__main__":
    main()
    