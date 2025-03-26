import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import calendar
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

#############################
#PARAMETERS
#############################
API_KEY_openweathermap = "f25f46afc61f305a6abbaeb8fd611fc0"  # Replace with a valid key
LAT, LON = 41.0082, 28.9784    # Istanbul coordinates
TRAIN_END_DATE = datetime(2024, 10, 31, 9, 0, tzinfo=timezone.utc)
TRAIN_START_DATE = TRAIN_END_DATE - timedelta(days=370)  # ~2 years back
TEST_START_DATE = datetime(2024, 10, 29, 9, 0, tzinfo=timezone.utc)
TEST_END_DATE = datetime(2024, 11, 30, 9, 0, tzinfo=timezone.utc)

OPENMETEO_TRAIN_START_DATE = datetime(2000,1,1)
OPENMETEO_TRAIN_END_DATE = datetime(2024,10,31)
OPENMETEO_TEST_START_DATE =datetime(2024,10,29)
OPENMETEO_TEST_END_DATE = datetime(2024,11,30)

used_featurRes_OPENWEATHER =['temp', 'temp_max', 'humidity']
used_features_OPENMETEO = ['temperature_2m_mean', 'temperature_2m_max','temperature_2m_min','apparent_temperature_max','precipitation_sum', 'day_sin', 'day_cos']
fetch_train_data_flag = False
fetch_test_data_flag = False
train_new_model = False
#############################
# FUNCTIONS
#############################
def fetch_weather_data(start_date, end_date, lat, lon, api_key):
    """Fetch daily weather data from OpenWeather and return a DataFrame."""
    URL = "http://history.openweathermap.org/data/2.5/history/city"
    weather_data = []
    current_date = start_date
    while current_date <= end_date:
        unix_timestamp_start = current_date.timestamp()
        unix_timestamp_end = (current_date + timedelta(days=1)).timestamp()
        params = {
            "lat": lat,
            "lon": lon,
            "start": unix_timestamp_start,
            "end": unix_timestamp_end,
            "appid": api_key,
            "units": "metric"
        }
        try:
            response = requests.get(URL, params=params)
            response.raise_for_status()
            data = response.json()
            for record in data.get("list", []):
                weather_data.append({
                    "date": datetime.fromtimestamp(record["dt"], tz=timezone.utc).strftime('%Y-%m-%d'),
                    "temp": record["main"]["temp"],
                    "feels_like": record["main"]["feels_like"],
                    "humidity": record["main"]["humidity"],
                    "pressure": record["main"]["pressure"],
                    "temp_max": record["main"]["temp_max"],
                    "temp_min": record["main"]["temp_min"]
                })
        except Exception as e:
            print(f"Failed to fetch data for {current_date.strftime('%Y-%m-%d')}: {e}")
        current_date += timedelta(days=1)
    df = pd.DataFrame(weather_data)
    return df

def fetch_data_openmeteo(start_date, end_date, lat, lon, batch_size_days=365, ):
    current_date = start_date
    end_date_dt = end_date
    all_data = []

    while current_date <= end_date_dt:
        batch_end_date = current_date + timedelta(days=batch_size_days - 1)
        if batch_end_date > end_date_dt:
            batch_end_date = end_date_dt

        # Fetch data for the batch
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": current_date.strftime("%Y-%m-%d"),
            "end_date": batch_end_date.strftime("%Y-%m-%d"),
            "daily": ['temperature_2m_mean','temperature_2m_max', 'temperature_2m_min','apparent_temperature_max','apparent_temperature_min','precipitation_sum' ],
            "timeformat": "unixtime",
            "timezone": "auto"
        }
        response = requests.get("https://archive-api.open-meteo.com/v1/archive?", params=params)
        if response.status_code == 200:
            batch_data = response.json().get("daily", {})
            all_data.extend(pd.DataFrame(batch_data).to_dict(orient='records'))
        else:
            print(f"Error fetching data for {current_date} to {batch_end_date}: {response.status_code}")

        current_date = batch_end_date + timedelta(days=1)

    return pd.DataFrame(all_data)




#############################
# STEP 1: FETCH TRAIN/VAL DATA
#############################
#OPENMETEO Fetch code
if((not os.path.exists("OPENMETEO_historical_weather_data.csv")) or fetch_train_data_flag):
    df = fetch_data_openmeteo(OPENMETEO_TRAIN_START_DATE, OPENMETEO_TRAIN_END_DATE,LAT,LON)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.date
    df = df.rename(columns={'time': 'date'})
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['date']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['date']).dt.month / 12)
    df['day_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['date']).dt.day / 31)
    df['day_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['date']).dt.day / 31)
    df.to_csv("OPENMETEO_historical_weather_data.csv", index=False)
    print("All data saved to 'OPENMETEO_historical_weather_data.csv'.")

# Code to obtain data from openweather, not in use since openmeteo goes further back which resulted in better performance
"""
df_train_val = fetch_weather_data(TRAIN_START_DATE, TRAIN_END_DATE, LAT, LON, API_KEY_openweathermap)

df_train_val.to_csv("daily_weather_data_train_val.csv", index=False)
print("Weather data collected and saved as 'daily_weather_data_train_val.csv'.")
if df_train_val.empty:
    raise ValueError("No training/validation data fetched. Check API KEY or date ranges.")
"""

df_t_V = pd.read_csv("OPENMETEO_historical_weather_data.csv")

# Fit scaler on train/val data only
scaler = MinMaxScaler()
train_val_scaled = scaler.fit_transform(df_t_V[used_features_OPENMETEO])


X, y = [], []
for i in range(len(train_val_scaled) - 3):
    X.append(train_val_scaled[i:i+3])  # Previous 3 days
    y.append(train_val_scaled[i+3][0])  # Next day's temperature (scaled)

X, y = np.array(X), np.array(y)

# Split into training and testing sets (80% train, 20% test)
split_index = int(len(X) * 0.80)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]


#############################
# STEP 2: TRAIN THE MODEL
#############################
tf.random.set_seed(42)
np.random.seed(42)
print(X.shape)

checkpoint_path = "training_1/cp.ckpt.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,  verbose=1)
model = Sequential([
    LSTM(50, activation='relu', input_shape=(3, X.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
if((not os.path.exists(checkpoint_path)) or train_new_model):
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stop,cp_callback])
else:
    model.load_weights(checkpoint_path)


#############################
# STEP 3: FETCH TEST DATA (NOVEMBER 2024)
#############################

#Openweather test data fetching, not in use but I left it in case there was a need to test
"""print("Fetching test data (November 2024)...")
df_test = fetch_weather_data(TEST_START_DATE, TEST_END_DATE, LAT, LON, API_KEY_openweathermap)
df_test.to_csv("daily_weather_data_test.csv", index=False)
if df_test.empty:
    raise ValueError("No test data for November 2024. Check API or data availability.")"""


#OPENMETEO FETCH TEST, Since saved not fetched every time
if((not os.path.exists("OPENMETEO_TEST_weather_data.csv")) or fetch_test_data_flag):
    df = fetch_data_openmeteo(OPENMETEO_TEST_START_DATE, OPENMETEO_TEST_END_DATE,LAT,LON)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.date
    df = df.rename(columns={'time': 'date'})
    df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['date']).dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['date']).dt.month / 12)
    df['day_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['date']).dt.day / 31)
    df['day_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['date']).dt.day / 31)
    df.to_csv("OPENMETEO_TEST_weather_data.csv", index=False)
    print("All data saved to 'OPENMETEO_TEST_weather_data.csv'.")

test_data = pd.read_csv("OPENMETEO_TEST_weather_data.csv")

# Scale test data using the already fitted scaler
test_scaled = scaler.transform(test_data[used_features_OPENMETEO])

X_test, y_test = [], []
for i in range(len(test_scaled) - 3):
    X_test.append(train_val_scaled[i:i+3])  # Previous 3 days
    y_test.append(train_val_scaled[i+3][0])  # Next day's temperature (scaled)

X_test, y_test = np.array(X_test), np.array(y_test)

print("Test samples:", len(X_test))

#############################
# STEP 4: PREDICT ON TEST SET (NOVEMBER)
#############################
y_test_pred = model.predict(X_test).flatten()

# Inverse transform predictions and true values to use in error calculation
dummy_pred = np.zeros((len(y_test_pred), len(used_features_OPENMETEO)))
dummy_true = np.zeros((len(y_test), len(used_features_OPENMETEO)))

dummy_pred[:,0] = y_test_pred
dummy_true[:,0] = y_test

inv_pred = scaler.inverse_transform(dummy_pred)[:,0]
inv_true = scaler.inverse_transform(dummy_true)[:,0]
inv_errors = np.abs(inv_true - inv_pred)

print("Inverse-transformed errors for each November 2024 day:")
print(inv_errors)

# Print metrics to evaluate the model
mse = mean_squared_error(inv_true, inv_pred)
mae = mean_absolute_error(inv_true, inv_pred)
r2 = r2_score(inv_true, inv_pred)
print("November 2024 - MSE:", mse)
print("November 2024 - MAE:", mae)
print("November 2024 - R²:", r2)

def permutation_feature_importance(model, X, y, metric=mean_squared_error):
    """
    Compute permutation feature importance for a model and dataset.
    
    Parameters:
    - model: trained model (Keras, TensorFlow, etc.)
    - X: numpy array of shape (n_samples, time_steps, n_features)
    - y: true labels (n_samples,)
    - metric: function to evaluate model performance (default: mean_squared_error)
    
    Returns:
    - importance: dict mapping feature index to increase in error when that feature is permuted
    """
    
    # Get the baseline error
    baseline_predictions = model.predict(X)
    baseline_score = metric(y, baseline_predictions.flatten())
    
    importance = {}
    
    # Iterate over each feature
    for feature_idx in range(X.shape[2]):
        # Copy X to avoid altering the original test data
        X_permuted = X.copy()
        
        # Shuffle the values of the current feature across the samples
        np.random.shuffle(X_permuted[:, :, feature_idx])
        
        # Evaluate the model on the permuted dataset
        permuted_predictions = model.predict(X_permuted)
        permuted_score = metric(y, permuted_predictions.flatten())
        
        # The increase in error is the feature importance
        importance[feature_idx] = permuted_score - baseline_score
    
    return importance

# Assuming your model is trained and you have X_test, y_test
feature_importances = permutation_feature_importance(model, X_test, y_test)
print("Permutation Feature Importances:")
for idx, imp in feature_importances.items():
    print(f"Feature {idx}: {imp}")
#############################
# STEP 6: CALENDAR VISUALIZATION FOR NOVEMBER ERRORS
#############################
year = 2024
month = 11
nov_current = datetime(2024, 11 ,1)
nov_30= datetime(2024, 11 ,30)

y_test_dates = []
while nov_current <= nov_30:
    y_test_dates.append(nov_current)
    nov_current += timedelta(days=1)
day_error_dict = {}
for d, err in zip(y_test_dates, inv_errors):
    d_py = pd.to_datetime(d).date()
    if d_py.year == year and d_py.month == month:
        day_error_dict[d_py] = err

cal = calendar.Calendar(firstweekday=0)  # Monday=0
month_matrix = cal.monthdatescalendar(year, month)

errors_map = []
for week in month_matrix:
    week_errors = []
    for day in week:
        if day.month == month:
            day_err = day_error_dict.get(day, np.nan)
        else:
            day_err = np.nan
        week_errors.append(day_err)
    errors_map.append(week_errors)

errors_map = np.array(errors_map)

fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.imshow(errors_map, cmap='coolwarm', interpolation='nearest', aspect='auto')
plt.title("November 2024 Daily Prediction Errors")
fig.colorbar(cax, label='Error (°C)')

for i, week in enumerate(month_matrix):
    for j, day in enumerate(week):
        if day.month == month:
            val = errors_map[i, j]
            day_num = day.day
            if not np.isnan(val):
                ax.text(j, i, f"{day_num}\n{val:.2f}", ha='center', va='center', color='black')
            else:
                ax.text(j, i, str(day_num), ha='center', va='center', color='grey')

ax.set_xticks(range(7))
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_yticks(range(errors_map.shape[0]))
ax.set_yticklabels([f'Week {i+1}' for i in range(errors_map.shape[0])])

plt.tight_layout()
plt.show()
