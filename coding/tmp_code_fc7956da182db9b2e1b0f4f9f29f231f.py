import pandas as pd
import matplotlib.pyplot as plt

# Get the stock price data for NVDA and TESLA
nvda_data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/NVDA?period1=1577836800&period2=1609459199&interval=1d&events=history')
tesla_data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/TSLA?period1=1577836800&period2=1609459199&interval=1d&events=history')

# Select only the required columns (Date and Close price)
nvda_data = nvda_data[['Date', 'Close']]
tesla_data = tesla_data[['Date', 'Close']]

# Set the Date column as the index
nvda_data['Date'] = pd.to_datetime(nvda_data['Date'])
tesla_data['Date'] = pd.to_datetime(tesla_data['Date'])
nvda_data.set_index('Date', inplace=True)
tesla_data.set_index('Date', inplace=True)

# Calculate the correlation between NVDA and TESLA stock prices
correlation = nvda_data['Close'].corr(tesla_data['Close'])

# Plot the stock price change YTD for NVDA and TESLA
plt.figure(figsize=(10, 5))
plt.plot(nvda_data.index, nvda_data['Close'], label='NVDA')
plt.plot(tesla_data.index, tesla_data['Close'], label='TESLA')
plt.title('Stock Price Change YTD')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

print("Correlation between NVDA and TESLA stock prices: ", correlation)