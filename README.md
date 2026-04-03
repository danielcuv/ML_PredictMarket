# NVDA Auto Trader 📈
Sistema automatizado de trading usando Machine Learning y Alpaca API.

Predice el precio de cierre del día siguiente de NVDA con un Random Forest Regressor y ejecuta órdenes de compra/venta automáticamente en paper trading.

---

## ¿Cómo funciona?

1. Descarga 3 años de datos históricos de NVDA vía Alpaca API
2. Construye features técnicos: retorno diario, volatilidad, medias móviles
3. Entrena un Random Forest para predecir el precio de cierre del día siguiente
4. Genera una señal BUY / SELL / HOLD según el cambio predicho
5. Ejecuta la orden automáticamente en tu cuenta de paper trading

---

## Señales de trading

| Condición | Señal |
|---|---|
| Modelo predice subida ≥ +0.5% | BUY |
| Modelo predice bajada ≤ -0.5% | SELL |
| Cambio entre -0.5% y +0.5% | HOLD |

---

## Tech Stack

- Python 3.9
- scikit-learn — Random Forest Regressor
- alpaca-py — datos históricos + ejecución de órdenes
- pandas / numpy

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/nvda-auto-trader.git
cd nvda-auto-trader

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install alpaca-py scikit-learn pandas numpy
```

---

## Configuración

Crea una cuenta gratuita en [alpaca.markets](https://alpaca.markets) y genera tus API keys de paper trading.

Luego configura tus keys como variables de entorno:

```bash
export ALPACA_API_KEY="tu_api_key"
export ALPACA_SECRET_KEY="tu_secret_key"
```

O reemplázalas directamente en el archivo (solo para uso local, nunca las subas a GitHub):

```python
API_KEY    = "tu_api_key"
API_SECRET = "tu_secret_key"
```

---

## Uso

Corre el script cada mañana antes de que abra el mercado (9:30 AM CST):

```bash
python ML_PredictMarket.py
```

Output esperado:
```
NVDA Auto Trader - 2026-03-30 08:45
Threshold compra: +0.5%  |  Threshold venta: -0.5%

Estado de la cuenta (Paper Trading)
Cash disponible:   $100,000.00
Valor portafolio:  $100,000.00
Poder adquisitivo: $200,000.00

Descargando datos historicos de NVDA...
Descargados 750 datos historicos de NVDA
Entrenando el modelo...
    MAE del modelo: $5.20

Señal Generada:  BUY
Precio Actual:   $171.22
Precio Predicho: $176.78
Cambio Esperado: 3.25%
Posicion Actual: 0 acciones

Orden COMPRA ejecutada | 584 acciones de NVDA (~$99,992.48)
```

---

## Métricas del modelo

- MAE promedio: ~$3-6 sobre precio ~$170
- Dataset: 750 barras diarias (3 años)
- Features: close, return, volatility, ma_5, ma_10, ma_20
- Split: 80% train / 20% test (cronológico, sin shuffle)

---

## Rutina diaria recomendada

```
8:30 AM CST  →  Correr el script
9:30 AM CST  →  Mercado abre, orden se ejecuta
4:00 PM CST  →  Revisar resultado en dashboard de Alpaca
```

---

## Próximos pasos

- [ ] Agregar indicadores técnicos: RSI, MACD, Bollinger Bands
- [ ] Reemplazar Random Forest con LSTM
- [ ] Implementar backtester histórico
- [ ] Agregar stop loss dinámico

---

## Disclaimer

Este proyecto es únicamente para fines educativos. No constituye asesoramiento financiero. Usa paper trading (dinero simulado) para pruebas.
