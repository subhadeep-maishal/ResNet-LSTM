import xarray as xr
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
netcdf_file = r"/scratch/20cl91p02/ANN_BIO/RsNet/ann_input_data.nc"
ds = xr.open_dataset(netcdf_file)

# Extract data variables
fe = ds['fe'].values  # (time, depth, lat, lon)
po4 = ds['po4'].values
si = ds['si'].values
no3 = ds['no3'].values  # Predictor
nppv = ds['nppv'].values  # Target variable

# Extract latitude and longitude
latitude = ds['latitude'].values  # Shape: (lat,)
longitude = ds['longitude'].values  # Shape: (lon,)

# Since depth is constant, discard the depth dimension and focus on (time, lat, lon)
fe = fe[:, 0, :, :]
po4 = po4[:, 0, :, :]
si = si[:, 0, :, :]
no3 = no3[:, 0, :, :]
nppv = nppv[:, 0, :, :]  # Ensure this matches the structure

# Replace NaN values in predictors and target
fe = np.nan_to_num(fe, nan=np.nanmean(fe))
po4 = np.nan_to_num(po4, nan=np.nanmean(po4))
si = np.nan_to_num(si, nan=np.nanmean(si))
no3 = np.nan_to_num(no3, nan=np.nanmean(no3))
nppv = np.nan_to_num(nppv, nan=np.nanmean(nppv))

# Stack the input variables along a new channel dimension (fe, po4, si, no3)
inputs = np.stack([fe, po4, si, no3], axis=-1)  # Shape: (time, lat, lon, channels)

# Prepare input for LSTM
time_steps = 5  # Number of time steps to consider in each sequence
samples = inputs.shape[0] - time_steps
X_lstm = np.array([inputs[i:i + time_steps] for i in range(samples)])

# Target: Predict NPPV
y_lstm = nppv[time_steps:]  # Shape: (samples, lat, lon)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Normalize the data
scaler_X = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[2] * X_train.shape[3] * X_train.shape[4])
X_test_reshaped = X_test.reshape(-1, X_test.shape[2] * X_test.shape[3] * X_test.shape[4])
X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

scaler_y = StandardScaler()
y_train_reshaped = y_train.reshape(-1, y_train.shape[1] * y_train.shape[2])
y_test_reshaped = y_test.reshape(-1, y_test.shape[1] * y_test.shape[2])
y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_test_scaled = scaler_y.transform(y_test_reshaped).reshape(y_test.shape)

# Define the ResNet block as a custom layer
class ResNetBlockLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(ResNetBlockLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return tf.keras.layers.Add()([inputs, x])
    
    def compute_output_shape(self, input_shape):
        return input_shape  # Output shape is the same as input shape

# Define the ResNet + LSTM Model
inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(inputs)
x = tf.keras.layers.TimeDistributed(ResNetBlockLayer(filters=64))(x)  # Use the custom ResNet block layer
x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)  # Flatten spatial dimensions

# LSTM to process the temporal sequence of spatial features
x = tf.keras.layers.LSTM(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(y_train.shape[1] * y_train.shape[2])(x)  # Flattened output (lat * lon)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_scaled, y_train_scaled.reshape(y_train_scaled.shape[0], -1), 
                    epochs=50, batch_size=16, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(X_test_scaled, y_test_scaled.reshape(y_test_scaled.shape[0], -1))

# Make predictions
predictions = model.predict(X_test_scaled)
predicted_y = scaler_y.inverse_transform(predictions.reshape(-1, y_test.shape[1] * y_test.shape[2])).reshape(y_test.shape)

print("Test Loss:", test_loss)

# Compute average actual and predicted values across time steps
average_actual_nppv = np.nanmean(y_test, axis=0)  # Average across time dimension
average_predicted_nppv = np.nanmean(predicted_y, axis=0)

# Define output file path
output_file_path = r"/scratch/20cl91p02/ANN_BIO/RsNet/average_output_resnet+lstm_nppv.nc"

# Create a new NetCDF file
with xr.Dataset() as ds_out:
    ds_out.coords['latitude'] = ('latitude', latitude)
    ds_out.coords['longitude'] = ('longitude', longitude)
    ds_out['average_actual_nppv'] = (('latitude', 'longitude'), average_actual_nppv)
    ds_out['average_predicted_nppv'] = (('latitude', 'longitude'), average_predicted_nppv)
    ds_out.attrs['title'] = 'Average NPPV Concentrations (ResNet + LSTM)'
    ds_out.attrs['description'] = 'Contains average actual and predicted NPPV concentrations using predictors (fe, po4, si, no3)'
    ds_out['average_actual_nppv'].attrs['units'] = 'mg C m-2 d-1'
    ds_out['average_predicted_nppv'].attrs['units'] = 'mg C m-2 d-1'
    ds_out.to_netcdf(output_file_path)

print(f"Output saved to: {output_file_path}")
