import time
import requests
import pandas as pd
import matplotlib.pyplot as plt

def fetch_binance_data(symbol="BTCUSDT", interval="1m", total_limit=5000):
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    limit_per_request = 1000
    interval_ms = 60 * 1000 

    end_time = int(time.time() * 1000) 
    rounds = total_limit // limit_per_request

    for i in range(rounds):
        start_time = end_time - (limit_per_request * interval_ms)
        url = f"{base_url}?symbol={symbol}&interval={interval}&limit={limit_per_request}&startTime={start_time}&endTime={end_time}"
        response = requests.get(url)
        data = response.json()
        if not data:
            break
        all_data = data + all_data  
        end_time = start_time
        time.sleep(0.1) 

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    return df["close"].tolist(), df

def calculate_rsi_loop(close_prices, period=14):
    rsi_values = []
    for i in range(len(close_prices)):
        if i < period:
            rsi_values.append(None)
            continue
        gains = 0
        losses = 0
        for j in range(i - period + 1, i + 1):
            delta = close_prices[j] - close_prices[j - 1]
            if delta > 0:
                gains += delta
            else:
                losses -= delta
        avg_gain = gains / period
        avg_loss = losses / period if losses != 0 else 1e-10
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    return rsi_values

def calculate_rsi_vectorized(close_series, period=14):
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def benchmark(data_sizes):
    loop_times = []
    vec_times = []
    speed_ratios = []
    
    for size in data_sizes:
        print(f"\nBenchmarking for data size: {size}")
        close_list, df = fetch_binance_data(total_limit=size)
        
        # Loop-based RSI
        start = time.time()
        rsi_loop = calculate_rsi_loop(close_list)
        end = time.time()
        loop_time = (end - start) * 1000  # milliseconds
        loop_times.append(loop_time)
        
        # Vectorized RSI
        start = time.time()
        rsi_vec = calculate_rsi_vectorized(df['close'])
        end = time.time()
        vec_time = (end - start) * 1000  # milliseconds
        vec_times.append(vec_time)
        
        speed_ratio = loop_time / vec_time
        speed_ratios.append(speed_ratio)
        
        print(f"Size: {size} | Loop: {loop_time:.2f} ms | Vectorized: {vec_time:.2f} ms | Speed Ratio: {speed_ratio:.2f}x")
    
    return loop_times, vec_times, speed_ratios

def plot_results(data_sizes, loop_times, vec_times, speed_ratios):
    plt.figure(figsize=(15, 5))
    
    # Execution Time Plot
    plt.subplot(1, 2, 1)
    plt.plot(data_sizes, loop_times, 'o-', label='Loop-based')
    plt.plot(data_sizes, vec_times, 'o-', label='Vectorized')
    plt.xlabel('Data Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    
    # Speed Ratio Plot
    plt.subplot(1, 2, 2)
    plt.plot(data_sizes, speed_ratios, 'o-', color='green')
    plt.xlabel('Data Size')
    plt.ylabel('Speed Ratio (x times faster)')
    plt.title('Vectorized Speed Advantage')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with increasing data sizes
    data_sizes = [1000, 10000, 50000, 100000, 200000, 500000]
    
    loop_times, vec_times, speed_ratios = benchmark(data_sizes)
    plot_results(data_sizes, loop_times, vec_times, speed_ratios)
