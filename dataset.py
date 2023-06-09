#import libraries
import cryptowatch as cw
cw.api_key = "your API key here"
import pandas as pd
import pandas_ta as ta
from scipy.signal import savgol_filter
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

#get cryptocurrencies data
def get_crypto_data():
    dataset = {}
    kraken = cw.markets.list("kraken")
    for market in kraken.markets:
        ticker = "{}:{}".format(market.exchange, market.pair).upper()
        if ticker.endswith('USD'):  #use data only denominated in US dollars
            candles = cw.markets.get(ticker, ohlc=True, periods=["1d"])
            try:
                if len(candles.of_1d) > 1500:
                    dataset[ticker] = pd.DataFrame(candles.of_1d, columns = candles._legend)
            except:
                pass
    return dataset
dataset = get_crypto_data()

#get location features
def loc_feature(data):
    loc = []
    for i in range(1,len(data)):
        close1 = data["close"][i] # today
        high1 = data['high'][i]
        low1 = data['low'][i]
        volume1 = data['volume quote'][i]
        if data["volume base"][i] != 0:
            aver_traded_price1 = data["volume quote"][i] / data["volume base"][i]
        else:
            aver_traded_price1 = 0 


        close0 = data['close'][i-1] # yesterday
        high0 = data['high'][i-1]
        low0 = data['low'][i-1]
        volume0 = data['volume quote'][i-1]
        if data["volume base"][i-1] != 0:
            aver_traded_price0 = data["volume quote"][i-1] / data["volume base"][i-1]
        else:
            aver_traded_price0 = 0

        if close1 > close0: 
            if aver_traded_price1 > aver_traded_price0:
                if volume1 >  volume0:
                    if high1 > high0:
                        if low1 > low0:
                            loc.append(0)
                        else:
                            loc.append(1)
                    else:
                        if low1 > low0:
                            loc.append(2)
                        else:
                            loc.append(3)
                    
                else:
                    if high1 > high0:
                        if low1 > low0:
                            loc.append(4)
                        else:
                            loc.append(5)
                    else:
                        if low1 > low0:
                            loc.append(6)
                        else:
                            loc.append(7)
            else:
                if volume1 >  volume0:
                    if high1 > high0:
                        if low1 > low0:
                            loc.append(8)
                        else:
                            loc.append(9)
                    else:
                        if low1 > low0:
                            loc.append(10)
                        else:
                            loc.append(11)
                    
                else:
                    if high1 > high0:
                        if low1 > low0:
                            loc.append(12)
                        else:
                            loc.append(13)
                    else:
                        if low1 > low0:
                            loc.append(14)
                        else:
                            loc.append(15)
                    
        else: 
            if aver_traded_price1 > aver_traded_price0:
                if volume1 >  volume0:
                    if high1 > high0:
                        if low1 > low0:
                            loc.append(16)
                        else:
                            loc.append(17)
                    else:
                        if low1 > low0:
                            loc.append(18)
                        else:
                            loc.append(19)
                    
                else:
                    if high1 > high0:
                        if low1 > low0:
                            loc.append(20)
                        else:
                            loc.append(21)
                    else:
                        if low1 > low0:
                            loc.append(22)
                        else:
                            loc.append(23)
            else:
                if volume1 >  volume0:
                    if high1 > high0:
                        if low1 > low0:
                            loc.append(24)
                        else:
                            loc.append(25)
                    else:
                        if low1 > low0:
                            loc.append(26)
                        else:
                            loc.append(27)
                    
                else:
                    if high1 > high0:
                        if low1 > low0:
                            loc.append(28)
                        else:
                            loc.append(29)
                    else:
                        if low1 > low0:
                            loc.append(30)
                        else:
                            loc.append(31)
    loc = pd.DataFrame(loc, index = data.index[1:])
    data["Location"] = loc
    return data

#get rest of the features and output labels
def get_features(dataset):
    for ticker in dataset:
        dataset[ticker]['returns'] = dataset[ticker]['close'].pct_change()
        direction = dataset[ticker]['close'].pct_change().shift(-1)
        direction[direction.between(-0.01, 0.01)] = 0
        direction[direction > 0] = 1
        direction[direction < 0] = -1
        dataset[ticker]['direction'] = direction
        
        dataset[ticker]['close timestamp'] = pd.to_datetime(dataset[ticker]['close timestamp'], unit = 's')
        dataset[ticker].rename(columns = {'close timestamp' : 'date'}, inplace = True)
        
        dataset[ticker].reset_index(drop = True, inplace = True)
        dataset[ticker] = loc_feature(dataset[ticker])
        
        dataset[ticker]['ema5'] = ta.ema(dataset[ticker]['close'], length = 5)
        dataset[ticker]['roc5'] = ta.roc(dataset[ticker]['close'], length = 5)
        dataset[ticker]['cci5'] = ta.cci(dataset[ticker]['high'], dataset[ticker]['low'],
                                         dataset[ticker]['close'], length = 5)
        dataset[ticker]['eom5'] = ta.eom(dataset[ticker]['high'], dataset[ticker]['low'],
                                         dataset[ticker]['close'], dataset[ticker]['volume base'],
                                         length = 5)
        rolling_window = dataset[ticker]['returns'].rolling(window = 21, min_periods = 21)
        dataset[ticker]['skewness'] = rolling_window.skew()
        
        rolling_window_percentiles = 21
        a = dataset[ticker]["close"].rolling(window = rolling_window_percentiles, min_periods = 21).rank()
        b = dataset[ticker]["close"].rolling(window = rolling_window_percentiles, min_periods = 21).count()
        percentiles = a/b
        dataset[ticker]['percentiles'] = percentiles
        dataset[ticker].dropna(inplace = True)
        dataset[ticker]['filtered percentiles'] = savgol_filter(dataset[ticker]['percentiles'], rolling_window_percentiles, 1)
        dataset[ticker].drop(columns = ['percentiles'], inplace = True)
    return dataset
dataset = get_features(dataset)
dataset = pd.concat(dataset)
df_droplevel = dataset.droplevel(1)
dataset = df_droplevel.groupby(level = 0)

#Spearman Correlation
def get_corr_matrix(data):
    combined_matrix = np.zeros((8, 8))
    
    for group_name, group_df in data:
      
        group_data = group_df.iloc[:, [7,6,10,11,12,13,14,15]]
    
    
        corr_matrix, _ = spearmanr(group_data)
    
    
        combined_matrix += corr_matrix
    
    average_matrix = combined_matrix / 17
    correlation_df = pd.DataFrame(average_matrix, columns=df_droplevel.iloc[:,[7,6,10,11,12,13,14,15]].columns, index=df_droplevel.iloc[:,[7,6,10,11,12,13,14,15]].columns)
    return correlation_df
correlation_df = get_corr_matrix(dataset)

plt.figure(figsize=(16, 10),dpi = 600)
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f', linewidths=.5)
plt.title("Spearman Correlation Heatmap")
plt.xticks(rotation = 45)
plt.show()
# save your data
df_droplevel.to_csv(r"C:\Users\86189\Desktop\Quantitative Finance\dataSet2.csv")
