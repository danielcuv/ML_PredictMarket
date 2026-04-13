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

#configuracion del entorno

API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")

#simbolos que pondre

SYMBOLS = ["NVDA", "PLTR", "SPY", "GOOG"]
ACTIVE_SYMBOL = "NVDA"

BUY_THRESHOLD = 0.005
SELL_THRESHOLD = -0.005

INITIAL_CAPITAL = 100_000

#Exportar cliente

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper = True)


#Descargar los datos
def fetch_data(symbol: str, years: int = 3) -> pd.DataFrame:
    print(f"DEscargando los datos historicos de {symbol}...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days = 365 * years)

    request = StockBarsRequest(
        symbol_or_symbols = symbol,
        timeframe = TimeFrame.Day,
        start = start,
        end = end,
        feed = "iex",
    )

    bars = data_client.get_stock_bars(request)
    df = bars.df.reset_index()

    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()
    if "timestamp" in df.columns:
        df = df.rename(columns = {"timestamp": "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop = True)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    print(f"Descargados {len(df)} registros de {symbol}")
    return df

#indicadores tecnicos

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # RSI que es (Relative Strength Index)
    # Mide la magnitud y velocidad de los movimientos de precios
    # >70 sobrecomprado
    # <30 sobrevendido
    # FORMULA = Promedio Ganancias / Promeido de perdidas

    delta = series.diff()
    gain = delta.clip(lower = 0)
    loss = - delta.clip(upper = 0)

    avg_gain = gain.ewm(com = period - 1, min_periods = period).mean()
    avg_loss = loss.ewm(com = period - 1, min_periods = period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_macd(series: pd.Series,fast: int = 12, slow: int = 26, signal: int = 9):
    # MACD Moving Average COvergence Divergence
    # Mide la diferencia entre dos EMA's para detectar el moemntum y cambios de tendencia

    # Esto se compone de:
    # MACD Line = EMA(12) - EMA(26) -> velocidad de precio
    # Signal Line = EMA(9) de MACD Line -> Suavisado del MACD
    # Hitograma = MACD Line - Signal Line -> Fuerza de momentum

    # Las Señales Clave son:
    # MACD cruza Signal hacia arriba -> señal alcista
    # MACD cruza Signal hacia abajo -> señal bajista
    # Histograma positivo/creciente -> momentum alcista

    ema_fast = series.ewm(span = fast, adjust = False).mean()
    ema_slow = series.ewm(span = slow, adjust = False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span = signal, adjust = False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    # Bandas de Bollinger
    # Miden la volatilidad relativa del precio respecto a su media movil.

    # Sus componentes son:
    # Middle Band = SMA(20)
    # Upper Band = SMA(20) + 2 * σ(20)
    # Lower band = SMA(20) - 2 * σ(20)

    # Los indicadores de derivados son:
    # %B    = (precio - lower) / (upper - lower)
    #       > 1 = precio sobre banda superior
    #       < 0 precio bajo demanda inferior
    # Bandwidth = (upper - lower) / middle -> expansion / contraccion de la volatilidad

    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bwidth = (upper - lower) / middle
    return upper, middle, lower, pct_b, bwidth

# Contruccion de las caracteristicas (Features)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #Features originales
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(5).std()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    # RSI
    df["rsi"] = calc_rsi(df["close"], period  = 14)

    # MACD
    macd_line, signal_line, histogram = calc_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram

    #bollinger bansd ------ bandas de bollier
    upper, middle, lower, pct_b, bwidth = calc_bollinger_bands(df["close"])
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    df["bb_pct_b"] = pct_b
    df["bb_width"] = bwidth

    # el precio del siguiente dia (target)

    df["target"] = df["close"].shift(-1)

    df = df.dropna().reset_index(drop = True)
    return df

# Entrenamiento del Modelo

FEATURE_COLS = [
    # Muestra Precios y Tendencias
    "close", "return", "volatility",
    "ma_5", "ma_10", "ma_20",
    # Momentum
    "rsi", "macd",
    "macd_signal", "macd_histogram",
    # Volatilidad relativa
    "bb_pct_b", "bb_width"
]

def train_model(df: pd.DataFrame):
    print(" Entrenando el modelo...")
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, shuffle = False
    )
    model = RandomForestRegressor(
        n_estimators = 200,
        max_depth = 6,
        min_samples_leaf = 5,
        random_state = 42,
        n_jobs = -1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"    MAE del modelo: ${mae:.2f}")

    #IMportancias en los Features

    importances = pd.Series(model.feature_importances_, index = FEATURE_COLS)
    top5 = importances.nlargest(5)
    print(" Top - 5 features mas importantes: ")
    for feat, val in top5.items():
        print(f"    {feat:<22} {val:.4f}")

    return model

# 5. Señal para tradear

def get_signal(model, df: pd.DataFrame) -> tuple:
    latest = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    current_price = float(df["close"].iloc[-1])
    predicted_price = float(model.predict(latest)[0])
    pct_change = (predicted_price - current_price) / current_price

    if pct_change >= BUY_THRESHOLD:
        signal = "BUY"
    elif pct_change <= SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, current_price, predicted_price, pct_change

# Backtesting el modelo.

def run_backtest(df: pd.DataFrame, model, initial_capital: float = INITIAL_CAPITAL, symbol: str = "???") -> pd.DataFrame:
    # Backtester simple
    # Simula las señales del modelo en datos historicos.

    # la logica es la siguiente
    # -Entrena con el 70% de los datos y simula dia a dia con el 30% restante
    # -BUY: invierte todo el capital disponible
    # -SELL: vende toda la posicion
    # -HOLD: no hace nada

    # metricas reportadas
    # Retorno total y anualizado
    # -Sharpe Ratio (retorno ajustado por riesgo)
    # -Max Drawdown (peor caida desde un maximo)
    # -Win Rate (%cantidad de operaciones ganadoras)
    #BUY & HOLD (benchmark: comprar y mantener)

    print(f"backtesting - {symbol}")

    split_idx = int(len(df) * 0.70)
    test_df = df.iloc[split_idx:].reset_index(drop = True)

    cash = initial_capital
    position = 0.0
    trades = []

    portfolio_values = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        current_price = row["close"]

        #Señal con lo que sbe el modelo hasta ese dia
        features_row = row[FEATURE_COLS].values.reshape(1, -1)
        predicted = float(model.predict(features_row)[0])
        pct_change = (predicted - current_price) / current_price

        if pct_change >= BUY_THRESHOLD:
            signal = "BUY"
        elif pct_change <= SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Ejecucion de la operacion
        if signal == "BUY" and cash > current_price:
            shares_bought = cash // current_price
            cash -= shares_bought * current_price
            position += shares_bought
            trades.append({
                "date": row["date"], "action": "BUY",
                "price": current_price, "shares": shares_bought,
            })

        elif signal == "SELL" and position > 0:
            cash += position * current_price
            trades.append({
                "date": row["date"], "action": "SELL",
                "price": current_price, "shares": position,
            })
            position = 0.0

        portfolio_value = cash + position * current_price
        portfolio_values.append({
            "date": row["date"],
            "price": current_price,
            "signal": signal,
            "cash": cash,
            "position_shares": position,
            "portfolio_value": portfolio_value,
        })

    results_df = pd.DataFrame(portfolio_values)

    # Metricas
    final_value = results_df["portfolio_value"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # Retorno diario para el SHARPE
    results_df["daily_return"] = results_df["portfolio_value"].pct_change()
    sharpe = (
        results_df["daily_return"].mean() /
        results_df["daily_return"].std() * np.sqrt(252)
        if results_df["daily_return"].std() > 0 else 0
    )

    # Max Drawdown
    rolling_max = results_df["portfolio_value"].cummax()
    drawdown = (results_df["portfolio_value"] - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Win Rate
    trades_df = pd.DataFrame(trades)
    win_rate = 0.0
    n_profitable = 0
    n_trades = 0
    if not trades_df.empty:
        buys = trades_df[trades_df["action"] == "BUY"].reset_index(drop = True)
        sells = trades_df[trades_df["action"] == "SELL"].reset_index(drop = True)
        pairs = min(len(buys), len(sells))
        n_trades = pairs
        if pairs > 0:
            buy_prices = buys["price"].iloc[:pairs].values
            sell_prices = sells["price"].iloc[:pairs].values
            n_profitable = int(np.sum(sell_prices > buy_prices))
            win_rate = n_profitable / pairs

    # Buy and Hold benchmark
    bh_return = (test_df["close"].iloc[-1] - test_df["close"].iloc[0]) / test_df["close"].iloc[0]
    # Dias que esta en el Test
    n_days = len(test_df)
    ann_ret = (1 + total_return) ** (252 / n_days) - 1

    #resumen del backtest
    print(f"    Periodo: {test_df['date'].iloc[0].date()} -> {test_df['date'].iloc[-1].date()}")
    print(f"    Capital Inicial: ${initial_capital:>12,.2f}")
    print(f"    Capital Final: ${final_value:>12,.2f}")
    print(f"    Retorno Total: {total_return * 100:>+.2f}%")
    print(f"    Retorno Anualiz: {ann_ret * 100:>+.2f}%")
    print(f"    Sharpe Ratio: {sharpe:> 8.3f}")
    print(f"    Max Drawdown: {max_dd * 100:>8.2f}%")
    print(f"    Operaciones: {n_trades}")
    print(f"    Win Rate: {win_rate * 100:>8.1f}%")
    print(f"    Buy & Hold: {bh_return * 100:>+.2f}% (benchmark)")

    if total_return > bh_return:
        print(f"    El modelo Supero al buy & Hold por {(total_return - bh_return) * 100:.2f}%")
    else:
        print(f"    El modelo quedó por debajo del Buy & Hold en {(bh_return - total_return)*100:.2f}%")

    return results_df

# Posicion y Ordenes de compra venta

def get_current_position(symbol: str) -> float:
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except Exception:
        return 0.0

def execute_order(signal: str, symbol: str, current_price: float):
    position = get_current_position(symbol)
    account = trading_client.get_account()
    cash = float(account.cash)

    if signal == "BUY":
        qty = int(cash // current_price)
        if qty <= 0:
            print(" No hay suficiente cash para comprar ni 1 accion")
            return
        order = MarketOrderRequest(
            symbol = symbol, qty = qty,
            side = OrderSide.BUY, time_in_force = TimeInForce.DAY,
        )
        result = trading_client.submit_order(order)
        print(f"    Orden de Compra ejecutada | {qty} acciones de {symbol} (~${qty * current_price:,.2f})")
        print(f"    ID: {result.id}")

    elif signal == "SELL":
        if position > 0:
            order = MarketOrderRequest(
                symbol = symbol, qty = int(position),
                side = OrderSide.SELL, time_in_force = TimeInForce.DAY,
            )
            result = trading_client.submit_order(order)
            print(f"    Orden VENTA ejecutada | {int(position)} acciones de {symbol}")
            print(f"    ID: {result.id}")
        else:
            print(f"    Señal Sell pero no tienes posicion en {symbol} - sin orden")
    else:
        print(" Señal HOLD - sin orden ejecutada")

def print_account_status():
    account = trading_client.get_account()
    print(f"\n  Estado de la cuenta (en Paper Trading)")
    print(f"    Cash disponible: ${float(account.cash):>12,.2f}")
    print(f"    Valor del Portafolio: ${float(account.portfolio_value):>12,.2f}")
    print(f"    Poder adquisitivo: ${float(account.buying_power):>12,.2f}\n")

# 8.MODO Multi-SIMBOLO

def scan_all_symbols():
    # esto corre todoel pipeline de los
    # simbolos y muestra las señales sin ejecutar ordenes
    print(f"    SCAN MULTI_SYMBOL - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    results = []
    for sym in SYMBOLS:
        try:
            df_raw = fetch_data(sym)
            df_feat = build_features(df_raw)
            model = train_model(df_feat)
            signal, cur, pred, pct = get_signal(model, df_feat)
            results.append({
                "Symbol": sym,
                "signal": signal,
                "Current": f"${cur:.2f}",
                "Pred": f"${pred:.2f}",
                "Δ%":      f"{pct*100:+.2f}%",
            })
        except Exception as e:
            results.append({"Symbol": sym, "Sygnal": "ERROR", "Current": "-", "Pred": "-", "Δ%": str(e)})

    summary = pd.DataFrame(results)
    print("\n" + summary.to_string(index = False))
    return summary

# Ejecucion MAIN

def main():
    print(f"    ML RandomForestRegressor - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"    simbolo activo: {ACTIVE_SYMBOL}")
    print(f"    threshold BUY: +{BUY_THRESHOLD * 100: .1f}%  |  SELL: {SELL_THRESHOLD * 100:.1f}%")

    print_account_status()

    #PIPELINE PRINCIPAL
    df_raw = fetch_data(ACTIVE_SYMBOL)
    df_feat = build_features(df_raw)
    model = train_model(df_feat)

    # CORRER EL BACKTEST ANTES DE OPERAR
    backtest_results = run_backtest(df_feat, model, initial_capital = INITIAL_CAPITAL, symbol = ACTIVE_SYMBOL)

    # Señal del dia
    print(f" Señal del dia")
    signal, current_price, predicted_price, pct_change = get_signal(model, df_feat)

    # RSI, MACD y Bollinger actuales para contexto
    rsi_now = float(df_feat["rsi"].iloc[-1])
    macd_now = float(df_feat["macd_histogram"].iloc[-1])
    bb_now = float(df_feat["bb_pct_b"].iloc[-1])

    print(f"    Señal: {signal}")
    print(f"    Precio Actual: ${current_price:.2f}")
    print(f"    Precio predicho: ${predicted_price:.2f}")
    print(f"    Cambio esperado: {pct_change * 100:+.2f}%")
    print(f"    RSI(14): {rsi_now:.1f} {'sobrecomprado' if rsi_now > 70 else 'sobrevendido' if rsi_now < 30 else 'neutro'}")
    print(f"    MACD Histogram: {macd_now:+.4f} {'alcista' if macd_now > 0 else 'bajista'}")
    print(f"    Bollinger %B:    {bb_now:.2f}  {'cerca banda sup.' if bb_now > 0.8 else '↓ cerca banda inf.' if bb_now < 0.2 else '-> zona media'}")
    print(f"    Posicion Actual: {get_current_position(ACTIVE_SYMBOL):.0f} acciones \n")

    # Ejecutar Orden
    print(" Ejecutando orden")
    execute_order(signal, ACTIVE_SYMBOL, current_price)

    print_account_status()


if __name__ == "__main__":
    main()
