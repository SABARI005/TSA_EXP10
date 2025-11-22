# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the e-commerce dataset
data = pd.read_csv('ECOMM DATA.csv')

# Convert the 'Order Date' to datetime and set it as the index
data['Order Date'] = pd.to_datetime(data['Order Date'])
data.set_index('Order Date', inplace=True)

# Ensure data is at daily frequency (resample if necessary)
data = data['Shipping Cost'].resample('D').sum().fillna(0)

# Plot the 'Shipping Cost' time series
plt.plot(data.index, data, label='Shipping Cost')
plt.xlabel('Date')
plt.ylabel('Shipping Cost ($)')
plt.title('Shipping Cost Time Series')
plt.show()

# Function to check stationarity of the 'Shipping Cost' time series
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data)

# Plot ACF and PACF for 'Shipping Cost' time series
plot_acf(data)
plt.show()
plot_pacf(data)
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Define and fit the SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot Actual vs Predicted 'Shipping Cost'
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Shipping Cost ($)')
plt.title('SARIMA Model Predictions for Shipping Cost')
plt.legend()
plt.xticks(rotation=45)
plt.show()
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/42a6a55e-81af-4b8b-a947-ad1f27605d51)
![image](https://github.com/user-attachments/assets/8542c1da-0982-4349-8576-4e11bf3c67e4)
![image](https://github.com/user-attachments/assets/b87458d6-7fd6-45a9-bcd3-6c1935ba6939)
![image](https://github.com/user-attachments/assets/efce0be0-89f8-42f1-9f5a-5d53485ff671)
![image](https://github.com/user-attachments/assets/3ebf836f-c6b2-4ba4-bf2b-608a7ad9b352)
![image](https://github.com/user-attachments/assets/3fc20f62-dd5a-4032-8d16-107bd3d5d89a)


### RESULT:
Thus the program run successfully based on the SARIMA model.
