# Setup for analysis
dataset = 1  # (1 - Liege, 2 - TUO, 3 - Rye)
horizon = 10
epochs = 100
data_split = 0.2 # Test (%)
epochs_h = 20
max_evals = 50
k_fold = 5
k_scalable = 20

save_images = True
fixed_samples = True

# General analysis
analysis_downsample = False
analysis_filters = False
analysis_horizon = False
analysis_hypertuning = False
analysis_opt_model_k_fold = False
analysis_statistics = False

# To evaluate our model step by step
analysis_our_model_steps = False
cnn_use = False
attention_use = False
filter_use_std = False 
filter_use_scalable = False

# For benchmarking
analysis_benchmarking = False
analysis_our_model = False
filter_use = False


if fixed_samples == True:
    # For a fixed data length
    data_length = 1000
else:
    # For one-month data length
    if dataset==1:
      data_length = 12*60*24*30 # Sample every 5 sec (12*60*24*30 = 1 month)
    if dataset==2:
      data_length = 60*24*30 # Sample every 1 min (60*24*30 = 1 month)
    if dataset==3:
      data_length = 24*30 # Sample every 1 h (24*30 = 1 month)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error

from scipy.signal import wiener
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.layers import Attention, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
###############################################################################
################################ Load Data ####################################
###############################################################################

# Time series preprocessing dataset 1
if dataset == 1:
  df = pd.read_csv('./Dataset/miris_load.csv', parse_dates=['DateTime'])
  df = df.rename(columns={'Conso': 'Load'})
  time_df = df['DateTime']
  df.set_index('DateTime', inplace=True)

  if save_images == True:
      plt.figure(figsize=(8, 3))
      plt.plot(df['Load'], 'k', zorder=2)
      plt.xlabel('Time')
      plt.ylabel('Load (kW)')
      plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      plt.savefig('./Results/original1.pdf', bbox_inches = 'tight')
      plt.show()

  preprocessed = df[0:data_length]  # sample every 5 sec (12*60*24*30 = 1 month)
  if save_images == True:
      plt.figure(figsize=(8, 3))
      plt.plot(preprocessed['Load'], 'k', zorder=2)
      plt.xlabel('Time')
      plt.ylabel('Load (kW)')
      plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      plt.savefig('./Results/preprocessed1.pdf', bbox_inches = 'tight')
      plt.show()

  preprocessed_d = df[0:12*60*24*3]
  p_downsample1 = preprocessed_d[::12]
  p_downsample2 = preprocessed_d[::12*30]
  p_downsample3 = preprocessed_d[::12*60]
  if save_images == True:
      plt.figure(figsize=(8, 3))
      plt.plot(p_downsample1['Load'], 'k', zorder=2)
      plt.xlabel('Time')
      plt.ylabel('Load (kW)')
      plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      plt.savefig('./Results/downsample1.pdf', bbox_inches = 'tight')
      plt.show()

  # Measure the period of each time series
  preprocessed_period = preprocessed.index[1]-preprocessed.index[0]
  preprocessed_period_sec = preprocessed_period.total_seconds()

# Time series preprocessing dataset 2
if dataset == 2:
  df = pd.read_excel('./Dataset/PowerQuality3-combin.xls', skiprows=2)
  df = df.rename(columns={'record time[s]': 'DateTime', 'avg.Pfh1[kW]': 'Load'})
  time_df = df['DateTime']
  df.set_index('DateTime', inplace=True)

  if save_images == True:
      plt.figure(figsize=(8, 3))
      plt.plot(df['Load'], 'k', zorder=2)
      plt.xlabel('Time')
      plt.xticks(rotation=45)
      plt.ylabel('Load (kW)')
      plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      plt.savefig('./Results/original2.pdf', bbox_inches = 'tight')
      plt.show()

  preprocessed=df[0:data_length] # sample every 1 min (60*24 = 1 day)
  if save_images == True:
      plt.figure(figsize=(8, 3))
      plt.plot(preprocessed['Load'], 'k', zorder=2)
      plt.xlabel('Time')
      plt.xticks(rotation=45)
      plt.ylabel('Load (kW)')
      plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      plt.savefig('./Results/preprocessed2.pdf', bbox_inches = 'tight')
      plt.show()

  # Measure the period of each time series
  preprocessed_period = preprocessed.index[1]-preprocessed.index[0]
  preprocessed_period_sec = preprocessed_period.total_seconds()

# Time series preprocessing dataset 3
if dataset == 3:
  df = pd.read_csv('./Dataset/rye_generation_and_load.csv', parse_dates=['index'])
  df = df.rename(columns={'index': 'DateTime', 'Consumption': 'Load'})
  time_df = df['DateTime']
  df.set_index('DateTime', inplace=True)

  if save_images == True:
      plt.figure(figsize=(8, 3))
      plt.plot(df['Load'], 'k', zorder=2)
      plt.xlabel('Time')
      plt.ylabel('Load (kW)')
      plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      plt.savefig('./Results/original3.pdf', bbox_inches = 'tight')
      plt.show()

  preprocessed=df[0:data_length] # sample every 1h (24 = 1 day)
  if save_images == True:
      plt.figure(figsize=(8, 3))
      plt.plot(preprocessed['Load'], 'k', zorder=2)
      plt.xlabel('Time')
      plt.xticks(rotation=45)
      plt.ylabel('Load (kW)')
      plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      plt.savefig('./Results/preprocessed3.pdf', bbox_inches = 'tight')
      plt.show()

  # Measure the period of each time series
  preprocessed_period = preprocessed.index[1]-preprocessed.index[0]
  preprocessed_period_sec = preprocessed_period.total_seconds()

###############################################################################
################################## Filter #####################################
###############################################################################
# Downsample filter analysis
if analysis_downsample == True:
  # Hodrick-Prescott Filter
  from statsmodels.tsa.filters.hp_filter import hpfilter

  # Apply the HP filter with a smoothing parameter lambda
  cycle, trend = hpfilter(preprocessed_d['Load'], lamb=1000)
  cycle, trend_1 = hpfilter(p_downsample1['Load'], lamb=1000)
  cycle, trend_2 = hpfilter(p_downsample2['Load'], lamb=1000)
  cycle, trend_3 = hpfilter(p_downsample3['Load'], lamb=1000)

  if save_images == True:
      # Plot the original and trend
      fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    
      ax1.plot(list(preprocessed_d['Load']), 'k', label='True Signal', linewidth=2)
      ax1.plot(list(trend), 'r', label='Trend', linewidth=2)
      ax1.set_ylim(0.04, 0.27)
      ax1.set_xlabel('Samples Index')
      ax1.set_ylabel('Load (kW)')
      ax1.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      ax1.set_title('(A)', loc='left')
      ax1.legend(loc='upper right')
    
      ax2.plot(list(p_downsample1['Load']), 'k', label='True Signal', linewidth=2)
      ax2.plot(list(trend_1), 'r', label='Trend', linewidth=2)
      ax2.set_ylim(0.04, 0.27)
      ax2.set_xlabel('Samples Index')
      ax2.set_ylabel('Load (kW)')
      ax2.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      ax2.set_title('(B)', loc='left')
      ax2.legend(loc='upper right')
    
      ax3.plot(list(p_downsample2['Load']), 'k', label='True Signal', linewidth=2)
      ax3.plot(list(trend_2), 'r', label='Trend', linewidth=2)
      ax3.set_ylim(0.04, 0.27)
      ax3.set_xlabel('Samples Index')
      ax3.set_ylabel('Load (kW)')
      ax3.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      ax3.set_title('(C)', loc='left')
      ax3.legend(loc='upper right')
    
      ax4.plot(list(p_downsample3['Load']), 'k', label='True Signal', linewidth=2)
      ax4.plot(list(trend_3), 'r', label='Trend', linewidth=2)
      ax4.set_ylim(0.04, 0.27)
      ax4.set_xlabel('Samples Index')
      ax4.set_ylabel('Load (kW)')
      ax4.grid(linestyle = '--', linewidth = 0.5, zorder=0)
      ax4.set_title('(D)', loc='left')
      ax4.legend(loc='upper right')
      plt.savefig('./Results/downsample.pdf', bbox_inches = 'tight')
      plt.show
      
# All filters analysis
if analysis_filters == True:
    from scipy.signal import savgol_filter, butter, filtfilt, medfilt, wiener
    from statsmodels.tsa.api import ExponentialSmoothing, STL
    from scipy.ndimage import gaussian_filter1d
    from pykalman import KalmanFilter
    
    Lambda = 3
    # Load time series
    ts = list(preprocessed['Load'])
    
    # 1. Simple Moving Average (SMA) filter
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    window_size = 15
    sma_filtered = moving_average(ts, window_size)
    
    # 2. Exponential Moving Average (EMA) filter
    def exponential_moving_average(data, alpha):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    alpha = Lambda/10  # Smoothing factor (0 < alpha < 1)
    ema_filtered = exponential_moving_average(ts, alpha)
    
    # 3. Savitzky-Golay filter (polynomial smoothing)
    savgol_window = Lambda*5
    savgol_order = Lambda
    savgol_filtered = savgol_filter(ts, savgol_window, savgol_order)
    
    # 4. Butterworth low-pass filter
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    fs = Lambda*10       # Sampling frequency
    cutoff = Lambda    # Desired cutoff frequency
    butter_filtered = butter_lowpass_filter(ts, cutoff, fs)
    
    # 5. Hodrick-Prescott filter (separates trend and cyclical components)
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cycle, hp_trend = hpfilter(ts, lamb=Lambda)
    
    # 6. Gaussian Filter
    gaussian_sigma = Lambda
    gaussian_filtered = gaussian_filter1d(ts, sigma=gaussian_sigma)
    
    # 7. Median Filter (non-linear)
    median_window = Lambda
    median_filtered = medfilt(ts, kernel_size=median_window)
    
    # 8. Wiener Filter 
    wiener_window = Lambda
    wiener_filtered = wiener(ts, mysize=wiener_window)
    
    # 9. Kalman Filter (state-space approach)
    def kalman_filter(observations, process_noise=0.1, measurement_noise=0.5):
        kf = KalmanFilter(
            initial_state_mean=observations[0],
            transition_matrices=[1],
            observation_matrices=[1],
            transition_covariance=process_noise,
            observation_covariance=measurement_noise
        )
        state_means, _ = kf.filter(observations)
        return state_means
    
    kalman_filtered = np.reshape(kalman_filter(ts), (1, data_length)).flatten()
    
    # 10. Wavelet Denoising (requires pywt)
    import pywt
    def wavelet_denoise(data, wavelet='db4', level=Lambda):
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-level])) / Lambda
        uthresh = sigma * np.sqrt(2*np.log(len(data)))
        coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
        return pywt.waverec(coeffs, wavelet)
    
    wavelet_filtered = wavelet_denoise(ts)
    
    # 11. Bandpass Filter (Butterworth)
    def butter_bandpass(lowcut, highcut, fs, order=Lambda):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    # 12. STL Decomposition (Seasonal-Trend decomposition)
    stl = STL(ts, period=Lambda)
    res = stl.fit()
    stl_trend = res.trend
        
    if save_images == True:
        # Plotting
        plt.figure(figsize=(8, 3))
        plt.plot(ts[500:1000], 'k--', label='True Signal', linewidth=2)
        
        # Filter plots
        plt.plot(sma_filtered[500:1000], label=f'SMA')
        plt.plot(stl_trend[500:1000], label='STL')
        plt.plot(ema_filtered[500:1000], label=f'EMA')
        plt.plot(savgol_filtered[500:1000], label=f'SG')
        plt.plot(hp_trend[500:1000], label='HP')
        plt.plot(butter_filtered[500:1000], label=f'Butterworth')
        plt.plot(gaussian_filtered[500:1000], label=f'Gaussian')
        plt.plot(median_filtered[500:1000], label=f'Median')
        plt.plot(wiener_filtered[500:1000], label=f'Wiener')
        plt.plot(kalman_filtered[500:1000], label='Kalman')
        plt.plot(wavelet_filtered[500:1000], 'r', label='Wavelet')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.legend()
        plt.xlabel('Samples Index')
        plt.ylabel('Load (kW)')
        plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
        plt.tight_layout()
        plt.show
        
        if dataset==1:
          plt.savefig('./Results/filters1.pdf', bbox_inches = 'tight')
        if dataset==2:
          plt.savefig('./Results/filters2.pdf', bbox_inches = 'tight')
        if dataset==3:
          plt.savefig('./Results/filters3.pdf', bbox_inches = 'tight')
      
    # RMSE & MAE & MAPE & MSLE & R2
    if dataset==1:
      print('\multirow{11}{*}{Liege}')
    if dataset==2:
      print('\multirow{11}{*}{TUO}')
    if dataset==3:
      print('\multirow{11}{*}{Rye}')
    
    y_true = list(preprocessed['Load'][0:len(sma_filtered)])
    y_pred = list(map(abs, sma_filtered))
    print(f'& SMA & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(abs(r2_score(y_true, y_pred))):.2E} //')
    y_true = list(preprocessed['Load'])
    y_pred = list(map(abs, stl_trend))
    print(f'& STL & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    y_pred = list(ema_filtered)
    print(f'& EMA & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    y_pred = list(map(abs, savgol_filtered))
    print(f'& SG & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    y_pred = list(map(abs, hp_trend))
    print(f'& HP & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    y_pred = list(map(abs, butter_filtered))
    print(f'& Butterworth & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    y_pred = list(map(abs, gaussian_filtered))
    print(f'& Gaussian & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    y_pred = list(map(abs, median_filtered))
    print(f'& Median & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    y_pred = list(map(abs, wiener_filtered))
    print(f'& Wiener & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    y_pred = list(map(abs, kalman_filtered))
    print(f'& Kalman & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    y_pred = list(map(abs, wavelet_filtered))
    print(f'& Wavelet & {(rmse(y_true, y_pred)):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')

# Scalable filter use
ts = list(preprocessed['Load'])
wiener_window = int(k_scalable/math.log(preprocessed_period_sec))
scalable_filter = wiener(ts, mysize=wiener_window)

if save_images == True:
    if dataset==1:
      color = 'r'
    if dataset==2:
      color = 'b'
    if dataset==3:
      color = 'g'
    
    fig, ((ax1)) = plt.subplots(1, 1, figsize=(4, 2), dpi=100)
    ax1.plot(list(preprocessed['Load'][500:1000]), 'k', label='True Signal', linewidth=2)
    ax1.plot(list(scalable_filter)[500:1000], color, label='Trend', linewidth=2)
    ax1.set_xlabel('Samples Index')
    ax1.set_ylabel('Load (kW)')
    ax1.grid(linestyle = '--', linewidth = 0.5, zorder=0)
    ax1.legend(loc='upper right')
    plt.tight_layout()
    plt.show
    if dataset==1:
      plt.savefig('./Results/scalable_filter1.pdf', bbox_inches = 'tight')
    if dataset==2:
      plt.savefig('./Results/scalable_filter2.pdf', bbox_inches = 'tight')
    if dataset==3:
      plt.savefig('./Results/scalable_filter3.pdf', bbox_inches = 'tight')    

###############################################################################
############################## Model Analysis #################################
###############################################################################

# Prepare the dataset
def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

###############################################################################
############################# Horizon Analysis ################################
###############################################################################
# Analysis of different horizon
if analysis_horizon == True:
    Results = []
    for look_back in range(5, 30, 5):
      # Load our dataset
      data_input = scalable_filter
                
      # Create DataFrame
      time_index = pd.to_datetime(time_df[0:len(data_input)])
      data = pd.DataFrame({'Load': data_input}, index=time_index)
    
      # Normalize the data
      scaler = MinMaxScaler(feature_range=(0, 1))
      data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
      X, y = create_dataset(data_scaled, look_back)
    
      # Reshape X to be compatible with Conv1D and LSTM layers
      X = X.reshape((X.shape[0], X.shape[1], 1))
    
      # Define the input shape
      input_shape = (look_back, 1)
    
      # Model setup
      inputs = Input(shape=input_shape) # Input layer
      x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs) # CNN Layer
      x = MaxPooling1D(pool_size=2)(x)
      x = LSTM(units=50, return_sequences=True)(x) # LSTM Layer
      attention = Attention()([x, x]) # Attention Layer
      x = Flatten()(attention) # Fully Connected Layer
      x = Dense(units=1)(x) # Define the model
      model = Model(inputs=inputs, outputs=x) # Compile the model
      model.compile(optimizer='adam', loss='mean_squared_error')
    
      # Train model
      model.fit(X, y, epochs=epochs, batch_size=32, validation_split=data_split)
      predicted = model.predict(X) # Predicting the next time step
      y_pred = list(map(abs, scaler.inverse_transform(predicted))) # There are no negative values 
      y_true = list(map(abs, scaler.inverse_transform(y.reshape(-1, 1)))) # Inverse transform the actual values
      # Save results
      Results.append(f'& {look_back} & {(rmse(y_true, y_pred)[0]):.2E} & {(mean_absolute_error(y_true, y_pred)):.2E} & {(mean_absolute_percentage_error(y_true, y_pred)):.2E} & {(mean_squared_log_error(y_true, y_pred)):.2E} & {(r2_score(y_true, y_pred)):.2E} //')
    
    # RMSE & MAE & MAPE & MSLE & R2
    if dataset==1:
      print('\multirow{5}{*}{Liege}')
    if dataset==2:
      print('\multirow{5}{*}{TUO}')
    if dataset==3:
      print('\multirow{5}{*}{Rye}')
    
    for i in range(0, len(Results)):
      print(Results[i])                                     
    
###############################################################################
########################### Hypertuning Analysis ##############################
###############################################################################
# Compute the model without hypertuning
if analysis_our_model_steps == True:
    def performance(y_true, y_pred, time_s):
        # Calculate metrics
        rmse_v = rmse(y_true, y_pred)[0]
        mae_v = mean_absolute_error(y_true, y_pred)
        mape_v = mean_absolute_percentage_error(y_true, y_pred)
        msle_v = mean_squared_log_error(y_true, abs(y_pred))
        r2_v = abs(r2_score(y_true, y_pred))
        # RMSE & MAE & MAPE & MSLE & R2 & time
        result = (f'{rmse_v:.2E} & {mae_v:.2E} & {mape_v:.2E} & {msle_v:.2E} & {r2_v:.2E} & {time_s:.2E} \\\\')
        return result

    if filter_use_std == True and filter_use_scalable == True:
        print('A filter need to be selected!'); data_input=[]
        
    elif filter_use_std == True and filter_use_scalable == False:
        data_input = wiener(list(preprocessed['Load']), mysize=3)
    elif filter_use_std == False and filter_use_scalable:
        data_input = scalable_filter
    else:
        data_input = preprocessed['Load']
        
    look_back = horizon   
    time_index = pd.to_datetime(time_df[0:len(data_input)])
    data = pd.DataFrame({'Load': data_input}, index=time_index)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = create_dataset(data_scaled, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    input_shape = (look_back, 1)
    inputs = Input(shape=input_shape) # Input layer
    if cnn_use == True: # CNN Layer
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    else:
        x = inputs
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(units=50, return_sequences=True)(x)  # LSTM Layers
    if attention_use == True:
        attention = Attention()([x, x]) # Attention Layer
        x = Flatten()(attention) # Fully Connected Layer
    else:
        x = Flatten()(x)
    x = Dense(units=1)(x) # Define the model
    model = Model(inputs=inputs, outputs=x) # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    start = time.time()
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=data_split)
    y_pred = scaler.inverse_transform(abs(model.predict(X)))
    y_true = scaler.inverse_transform(y.reshape(-1, 1))
    end = time.time()
    time_s = end - start
    performance_metrics = performance(y_true, y_pred, time_s)
    
    if cnn_use == False:
        if filter_use_std  == True and attention_use == True:
            F_LSTM_A = (f'F-LSTM-A & {performance_metrics}') 
            print(F_LSTM_A)
        elif filter_use_std  == True and attention_use == False:
            F_LSTM = (f'F-LSTM & {performance_metrics}') 
            print(F_LSTM)
        elif filter_use_scalable == True and filter_use_std  == False and attention_use == True:
            SF_LSTM_A = (f'SF-LSTM-A & {performance_metrics}') 
            print(SF_LSTM_A)
        elif filter_use_scalable == True and filter_use_std  == False and attention_use == False:
            SF_LSTM = (f'SF-LSTM & {performance_metrics}') 
            print(SF_LSTM)        
        elif filter_use_scalable == False and filter_use_std  == False and attention_use == True:
            LSTM_A = (f'LSTM-A & {performance_metrics}') 
            print(LSTM_A)
        elif filter_use_scalable == False and filter_use_std  == False and attention_use == False:
            LSTM = (f'LSTM & {performance_metrics}') 
            print(LSTM)
    
    elif cnn_use == True:
        if filter_use_std  == True and attention_use == True:
            F_CNN_LSTM_A = (f'F-CNN-LSTM-A & {performance_metrics}') 
            print(F_CNN_LSTM_A)
        elif filter_use_std  == True and attention_use == False:
            F_CNN_LSTM = (f'F-CNN-LSTM & {performance_metrics}') 
            print(F_CNN_LSTM)
        elif filter_use_scalable == True and filter_use_std  == False and attention_use == True:
            SF_CNN_LSTM_A = (f'SF-CNN-LSTM-A & {performance_metrics}') 
            print(SF_CNN_LSTM_A)
        elif filter_use_scalable == True and filter_use_std  == False and attention_use == False:
            SF_CNN_LSTM = (f'SF-CNN-LSTM & {performance_metrics}') 
            print(SF_CNN_LSTM)        
        elif filter_use_scalable == False and filter_use_std  == False and attention_use == True:
            CNN_LSTM_A = (f'CNN-LSTM-A & {performance_metrics}') 
            print(CNN_LSTM_A)
        elif filter_use_scalable == False and filter_use_std  == False and attention_use == False:
            CNN_LSTM = (f'CNN-LSTM & {performance_metrics}') 
            print(CNN_LSTM)
    
'''
print(LSTM)
print(LSTM_A)
print(F_LSTM)
print(F_LSTM_A)
print(SF_LSTM) 
print(SF_LSTM_A)

print(CNN_LSTM)
print(CNN_LSTM_A)
print(F_CNN_LSTM)
print(F_CNN_LSTM_A)
print(SF_CNN_LSTM) 
print(SF_CNN_LSTM_A)  
'''

# Compute hypertuning
if analysis_hypertuning == True:
    start = time.time()

    # Load our dataset
    look_back = horizon
    data_input = scalable_filter
    
    # Create DataFrame
    time_index = pd.to_datetime(time_df[0:len(data_input)])
    data = pd.DataFrame({'Load': data_input}, index=time_index)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = create_dataset(data_scaled, look_back)
    
    # Reshape X to be compatible with Conv1D and LSTM layers
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    from hyperopt import STATUS_OK
    def objective(params):
      FILTERS = int(params['filters'])
      LSTM_units = int(params['LSTM_units'])
      LSTM_layers = int(params['LSTM_layers'])
      OPT = str(params['optimizer'])
      BATCH_SIZE = int(params['batch_size'])
    
      input_shape = (look_back, 1)
      inputs = Input(shape=input_shape)
      x = Conv1D(filters=FILTERS, kernel_size=3, activation='relu')(inputs)
      x = MaxPooling1D(pool_size=2)(x)
    
      for _ in range(LSTM_layers):
        x = LSTM(units=LSTM_units, return_sequences=True)(x)
    
      attention = Attention()([x, x])
      x = Flatten()(attention)
      x = Dense(units=1)(x)
      model = Model(inputs=inputs, outputs=x)
      model.compile(optimizer=OPT, loss='mean_squared_error')
      model.fit(X, y, epochs=epochs_h, batch_size=BATCH_SIZE, validation_split=data_split)
      predicted = model.predict(X)
      predicted = scaler.inverse_transform(predicted)
      y_actual = scaler.inverse_transform(y.reshape(-1, 1))
    
      return {"loss": rmse(y_actual, predicted), "status": STATUS_OK}
    
    from hyperopt import hp
    
    space = {
        'LSTM_units': hp.quniform('LSTM_units', 10, 200, 10),
        'LSTM_layers': hp.quniform('LSTM_layers', 1, 10, 1),
        'optimizer': hp.choice('optimizer', ['sgd', 'rmsprop', 'adam', 'adamw', 'adadelta', 'adagrad', 'adamax', 'nadam']),
        'filters': hp.quniform('filters', 32, 1024, 32),
        'batch_size': hp.quniform('batch_size', 16, 256, 16),
    }
    
    from hyperopt import fmin, tpe, atpe, rand, anneal, Trials
    import warnings
    np.warnings = warnings
    
    # Store the results in Trials for later analysis
    trials = Trials()
    
    best = fmin(
        fn=objective,             # Objective function
        space=space,              # Hyperparameter space
        #algo=rand.suggest,       # Random Search
        #algo=anneal.suggest,     # Annealing Search
        #algo=tpe.suggest,        # Optimization algorithm TPE
        algo=atpe.suggest,
        max_evals=max_evals,      # Number of evaluations
        trials=trials             # Store results
    )
    
    from IPython.display import clear_output
    clear_output()
    
    from hyperopt import space_eval
    best_params = space_eval(space, best)
    print(best_params)
    
    # Extract loss values from each trial
    losses = [trial['result']['loss'] for trial in trials.trials]
    vals_LSTM_units = [trial['misc']['vals']['LSTM_units'][0] for trial in trials.trials]
    vals_LSTM_layers = [trial['misc']['vals']['LSTM_layers'][0] for trial in trials.trials]
    vals_optimizer = [trial['misc']['vals']['optimizer'][0] for trial in trials.trials]
    vals_filters = [trial['misc']['vals']['filters'][0] for trial in trials.trials]
    vals_batch_size = [trial['misc']['vals']['batch_size'][0] for trial in trials.trials]
    
    # Compute the cumulative minimum (best) loss
    best_losses = np.minimum.accumulate(losses)

    # Save losses of each optmization strategy
    #np.savetxt('./Results/best_losses_4.csv', best_losses, delimiter=',', fmt='%.10f')
    #np.savetxt('./Results/losses_4.csv', losses, delimiter=',', fmt='%.10f')
    
    if save_images == True:
        import matplotlib.tri as tri
        import seaborn as sns
   
        plt.figure(figsize=(7, 5))
        fig, (ax1) = plt.subplots(nrows=1)
        triang = tri.Triangulation(vals_LSTM_units, vals_LSTM_layers)
        interpolator = tri.LinearTriInterpolator(triang, losses)
        ax1.tricontour(vals_LSTM_units, vals_LSTM_layers, losses, levels=10, linewidths=0.5, colors='k')
        cntr2 = ax1.tricontourf(vals_LSTM_units, vals_LSTM_layers, losses, levels=30, cmap="RdBu_r")
    
        fig.colorbar(cntr2, ax=ax1, label='Loss')
        ax1.plot(vals_LSTM_units, vals_LSTM_layers, 'ko', ms=2)
        ax1.set_xlabel('LSTM units')
        ax1.set_ylabel('LSTM layers')
        plt.subplots_adjust(hspace=0.5)
        plt.savefig('./Results/opt_1.pdf', bbox_inches = 'tight')
        plt.show()
        
        # Combine data into a DataFrame
        data = pd.DataFrame({
            'LSTM_units': vals_LSTM_units,
            'LSTM_layers': vals_LSTM_layers,
            'optimizer': vals_optimizer,
            'filters': vals_filters,
            'batch_size': vals_batch_size,   
            'RMSE': losses
        })

        plt.figure(figsize=(7, 5))
        # Use seaborn pairplot
        sns.pairplot(data, vars=['LSTM_units', 'LSTM_layers', 'optimizer', 'filters', 'batch_size'], hue='RMSE', palette='viridis', diag_kind='kde')
        plt.savefig('./Results/opt_2.pdf', bbox_inches = 'tight')
        plt.show()
        
        from sklearn.decomposition import PCA
        from scipy.interpolate import griddata

        # Get the list of hyperparameter keys from the first trial
        keys = list(trials.trials[0]['misc']['vals'].keys())

        # Build a matrix where each row corresponds to a trial and each column is a hyperparameter value.
        # We assume each parameter value is stored as a one-element list.
        params_list = []
        for trial in trials.trials:
            trial_params = [trial['misc']['vals'][key][0] for key in keys]
            params_list.append(trial_params)

        params_array = np.array(params_list)
        losses = np.array([trial['result']['loss'] for trial in trials.trials])

        # Reduce all hyperparameter dimensions to 2D using PCA
        pca = PCA(n_components=2)
        params_2d = pca.fit_transform(params_array)

        # Create a grid over the 2D PCA space
        num_grid_points = 200
        grid_x, grid_y = np.mgrid[params_2d[:,0].min():params_2d[:,0].max():complex(num_grid_points),
                                  params_2d[:,1].min():params_2d[:,1].max():complex(num_grid_points)]

        # Interpolate the loss values onto the grid
        grid_z = griddata(params_2d, losses, (grid_x, grid_y), method='cubic')

        # Plot the background loss contour and overlay the trial points
        plt.figure(figsize=(7, 5))
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Loss')
        plt.scatter(params_2d[:, 0], params_2d[:, 1], c=losses, cmap='viridis', edgecolor='k')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        #plt.title('Hyperparameter Optimization Results (All Hyperparameters)')
        plt.savefig('./Results/opt_3.pdf', bbox_inches = 'tight')
        plt.show()

    end = time.time()
    time_s = end - start
    print(time_s)

    '''rand_suggest_b = pd.read_csv('./Results/best_losses_1.csv')
    anneal_suggest_b = pd.read_csv('./Results/best_losses_2.csv')
    tpe_suggest_b = pd.read_csv('./Results/best_losses_3.csv')
    atpe_suggest_b = pd.read_csv('./Results/best_losses_4.csv')
    
    rand_suggest = pd.read_csv('./Results/losses_1.csv')
    anneal_suggest = pd.read_csv('./Results/losses_2.csv')
    tpe_suggest = pd.read_csv('./Results/losses_3.csv')
    atpe_suggest = pd.read_csv('./Results/losses_4.csv')
    
    if save_images == True:
        # Plot the Best Loss per Trial
        plt.figure(figsize=(6, 3))
        plt.plot(rand_suggest_b, marker='o', linestyle='-', color='r', label='Rand suggest')
        plt.plot(anneal_suggest_b, marker='.', linestyle='-', color='b', label='Anneal suggest')
        plt.plot(tpe_suggest_b, marker='o', linestyle='-', color='g', label='TPE suggest')
        plt.plot(atpe_suggest_b, marker='.', linestyle='-', color='y', label='APTE suggest')
        plt.xlabel('Trial')
        plt.ylabel('Best loss')
        plt.legend()
        plt.grid()
        plt.savefig('./Results/opt_strategy_best.pdf', bbox_inches = 'tight')
        plt.show()
    
        # Plot the Loss per Trial
        plt.figure(figsize=(8, 3))
        plt.plot(rand_suggest, marker='o', linestyle='-', color='r', label='Rand suggest')
        plt.plot(anneal_suggest, marker='.', linestyle='-', color='b', label='Anneal suggest')
        plt.plot(tpe_suggest, marker='o', linestyle='-', color='g', label='TPE suggest')
        plt.plot(atpe_suggest, marker='.', linestyle='-', color='y', label='APTE suggest')
        plt.xlabel('Trial')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.savefig('./Results/opt_strategy.pdf', bbox_inches = 'tight')
        plt.show()'''
 
# Compute k_fold opt model
if analysis_opt_model_k_fold == True:
    # Hypertuned results
    LSTM_layers = 7; LSTM_units = 80; Batch_size = 16; 
    Filters = 384; Opt = 'adamw'

    # Load our dataset
    look_back = horizon
    data_input = scalable_filter
    
    # Create DataFrame
    time_index = pd.to_datetime(time_df[0:len(data_input)])
    data = pd.DataFrame({'Load': data_input}, index=time_index)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = create_dataset(data_scaled, look_back)
    
    # Reshape X to be compatible with Conv1D and LSTM layers
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Define the input shape
    input_shape = (look_back, 1)
    
    # Model setup
    inputs = Input(shape=input_shape) # Input layer
    x = Conv1D(filters=Filters, kernel_size=3, activation='relu')(inputs) # CNN Layer
    x = MaxPooling1D(pool_size=2)(x)
    # Add LSTM layers in a loop
    for _ in range(LSTM_layers):
        x = LSTM(units=LSTM_units, return_sequences=True)(x)  # LSTM Layer
    attention = Attention()([x, x]) # Attention Layer
    x = Flatten()(attention) # Fully Connected Layer
    x = Dense(units=1)(x) # Define the model
    model = Model(inputs=inputs, outputs=x) # Compile the model
    model.compile(optimizer=Opt, loss='mean_squared_error')
    
    from sklearn.model_selection import KFold

    # Train model
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=50)
    
    # Initialize array to store the performance metrics
    RMSE, MAE, MAPE, MSLE, R2 = [], [], [], [], []

    # Iterate over each fold
    for train_index, test_index in kf.split(X):
    
        # Split the data into training and testing sets
        X_train, X_true = X[train_index], X[test_index]
        y_train, y_true = y[train_index], y[test_index]
    
        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=Batch_size, validation_split=data_split)
    
        # Evaluate the model
        y_pred = scaler.inverse_transform(abs(model.predict(X_true)))
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
        
        RMSE.append(float(rmse(y_true, y_pred)))
        MAE.append(mean_absolute_error(y_true, y_pred))
        MAPE.append(mean_absolute_percentage_error(y_true, y_pred))
        MSLE.append(mean_squared_log_error(y_true, y_pred))
        R2.append(abs(r2_score(y_true, y_pred)))
            
    np.savetxt('./Statistics/RMSE_'+str(k_fold)+'_fold_dataset_'+str(dataset)+'.csv', RMSE, delimiter=',', fmt='%.10f')
    np.savetxt('./Statistics/MAE_'+str(k_fold)+'_fold_dataset_'+str(dataset)+'.csv', MAE, delimiter=',', fmt='%.10f')
    np.savetxt('./Statistics/MAPE_'+str(k_fold)+'_fold_dataset_'+str(dataset)+'.csv', MAPE, delimiter=',', fmt='%.10f')
    np.savetxt('./Statistics/MSLE_'+str(k_fold)+'_fold_dataset_'+str(dataset)+'.csv', MSLE, delimiter=',', fmt='%.10f')
    np.savetxt('./Statistics/R2_'+str(k_fold)+'_fold_dataset_'+str(dataset)+'.csv', R2, delimiter=',', fmt='%.10f')
    
if analysis_statistics == True:
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    k5_1 = pd.read_csv("./Statistics/RMSE_5_fold_dataset_1.csv", header=None)
    k5_2 = pd.read_csv("./Statistics/RMSE_5_fold_dataset_2.csv", header=None)
    k5_3 = pd.read_csv("./Statistics/RMSE_5_fold_dataset_3.csv", header=None)
    k10_1 = pd.read_csv("./Statistics/RMSE_10_fold_dataset_1.csv", header=None)
    k10_2 = pd.read_csv("./Statistics/RMSE_10_fold_dataset_2.csv", header=None)
    k10_3 = pd.read_csv("./Statistics/RMSE_10_fold_dataset_3.csv", header=None)
    k20_1 = pd.read_csv("./Statistics/RMSE_20_fold_dataset_1.csv", header=None)
    k20_2 = pd.read_csv("./Statistics/RMSE_20_fold_dataset_2.csv", header=None)
    k20_3 = pd.read_csv("./Statistics/RMSE_20_fold_dataset_3.csv", header=None)
    
    # Combine the data into a single DataFrame
    data = pd.concat([k5_1, k10_1, k20_1, k5_2, k10_2, k20_2, k5_3, k10_3, k20_3], axis=1)
    columns = ['Liege (k=5)', 'Liege (k=10)', 'Liege (k=20)', 'TUO (k=5)', 'TUO (k=10)', 'TUO (k=20)', 'Rye (k=5)', 'Rye (k=10)', 'Rye (k=20)']
    data.columns = columns
    dfphi = pd.DataFrame({'Liege (k=5)': data.iloc[:, 0], 'Liege (k=10)': data.iloc[:, 1], 'Liege (k=20)': data.iloc[:, 2], 'TUO (k=5)': data.iloc[:, 3], 'TUO (k=10)': data.iloc[:, 4], 'TUO (k=20)': data.iloc[:, 5], 'Rye (k=5)': data.iloc[:, 6], 'Rye (k=10)': data.iloc[:, 7], 'Rye (k=20)': data.iloc[:, 8]})

    # calculate statistics
    summary_df = pd.DataFrame({
        'Mean': dfphi.mean(),
        'Median': dfphi.median(),
        'Mode': dfphi.mode().iloc[0],
        'Range': dfphi.max() - dfphi.min(),
        'Variance': dfphi.var(),
        'Std. Dev.': dfphi.std(),
        '25th %ile': dfphi.quantile(0.25),
        '50th %ile': dfphi.quantile(0.50),
        '75th %ile': dfphi.quantile(0.75),
        'IQR': dfphi.quantile(0.75) - dfphi.quantile(0.25),
        'Skewness': dfphi.skew(),
        'Kurtosis': dfphi.kurtosis()
    })
    
    # print summary to latex
    print(summary_df.transpose().round(5).to_latex(float_format="{:.1e}".format))

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create subplots without sharing y-axis
    fig, axes = plt.subplots(1, 9, figsize=(12, 6))
    
    # Plotting
    for i, col in enumerate(dfphi.columns):
        sns.boxplot(y=dfphi[col], ax=axes[i], color='c')
        plt.grid(linestyle='-', which='both')
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel('')  # Remove individual y-axis labels
        axes[i].grid(True)  # Enable gridlines for each subplot
    
    # Add a single y-axis label to the figure
    fig.text(0.04, 0.5, 'RMSE', va='center', rotation='vertical')
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.savefig("./Results/stats.pdf", dpi=800, bbox_inches = 'tight')
    plt.show()

if analysis_our_model == True:
    def performance(y_true, y_pred, time_s):
        # Calculate metrics
        rmse_v = rmse(y_true, y_pred)[0]
        mae_v = mean_absolute_error(y_true, y_pred)
        mape_v = mean_absolute_percentage_error(y_true, y_pred)
        msle_v = mean_squared_log_error(y_true, abs(y_pred))
        r2_v = abs(r2_score(y_true, y_pred))
        # RMSE & MAE & MAPE & MSLE & R2 & time
        result = (f'{rmse_v:.2E} & {mae_v:.2E} & {mape_v:.2E} & {msle_v:.2E} & {r2_v:.2E} & {time_s:.2E} \\\\')
        return result
    # Hypertuned results
    LSTM_layers = 7; LSTM_units = 80; Batch_size = 16; Filters = 384; Opt = 'adamw'
    look_back = horizon
    data_input = scalable_filter
    time_index = pd.to_datetime(time_df[0:len(data_input)])
    data = pd.DataFrame({'Load': data_input}, index=time_index)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = create_dataset(data_scaled, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    input_shape = (look_back, 1)
    inputs = Input(shape=input_shape) # Input layer
    x = Conv1D(filters=Filters, kernel_size=3, activation='relu')(inputs) # CNN Layer
    x = MaxPooling1D(pool_size=2)(x)
    for _ in range(LSTM_layers):
        x = LSTM(units=LSTM_units, return_sequences=True)(x)  # LSTM Layers
    attention = Attention()([x, x]) # Attention Layer
    x = Flatten()(attention) # Fully Connected Layer
    x = Dense(units=1)(x) # Define the model
    model = Model(inputs=inputs, outputs=x) # Compile the model
    model.compile(optimizer=Opt, loss='mean_squared_error')
    start = time.time()
    model.fit(X, y, epochs=epochs, batch_size=Batch_size, validation_split=data_split)
    y_pred = scaler.inverse_transform(abs(model.predict(X)))
    y_true = scaler.inverse_transform(y.reshape(-1, 1))
    end = time.time()
    time_s = end - start
    performance_metrics = performance(y_true, y_pred, time_s)
    Our = (f'Our & {performance_metrics}') 
    print(Our)

if analysis_benchmarking == True:
    # Load libraries
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT
    from neuralforecast.models import DilatedRNN
    from neuralforecast.models import NHITS
    from neuralforecast.models import TCN
    from neuralforecast.models import LSTM
    from neuralforecast.models import RNN
    from neuralforecast.models import NBEATS
    from neuralforecast.models import NBEATSx
    from neuralforecast.models import GRU
    from neuralforecast.models import Informer
    from neuralforecast.models import TiDE
    from neuralforecast.models import PatchTST
    from neuralforecast.models import FEDformer
    from neuralforecast.models import MLP
    from neuralforecast.models import TimesNet
    from neuralforecast.models import DeepAR
    from neuralforecast.models import BiTCN
    
    # Model setup
    max_steps = epochs
    input_size = horizon  
    freq = 's'
    
    if filter_use  == True:
        x = wiener(list(preprocessed['Load']), mysize=3)
    else:
        x = preprocessed['Load']
     
    # Create the time data
    time_data = pd.date_range(str(time_df[0]), periods=len(x), freq=freq)
    
    # Create the DataFrame
    df = pd.DataFrame({
        'unique_id': np.array(['Airline1'] * len(x)),
        'ds': time_data,
        'y': x,
        'trend': np.arange(len(x)),})
    
    # Split data based on your criteria
    split_point = int(len(df) * (1-data_split))
    Y_train_df = df.iloc[:split_point].reset_index(drop=True)
    Y_test_df = df.iloc[split_point:].reset_index(drop=True)
    y_true = Y_test_df.y[:horizon]
    
    def performance(y_true, y_pred, time_s):
        # Calculate metrics
        rmse_v = rmse(y_true, y_pred)
        mae_v = mean_absolute_error(y_true, y_pred)
        mape_v = mean_absolute_percentage_error(y_true, y_pred)
        msle_v = mean_squared_log_error(y_true, abs(y_pred))
        # r2_v = abs(r2_score(y_true, y_pred))
        # RMSE & MAE & MAPE & MSLE & R2 & time
        result = (f'{rmse_v:.2E} & {mae_v:.2E} & {mape_v:.2E} & {msle_v:.2E} & {time_s:.2E} \\\\') # & {r2_v:.2E}
        return result

    start = time.time()
    models = [MLP(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.MLP[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    MLP = (f'MLP & {performance_metrics}') 

    start = time.time()
    models = [TFT(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.TFT[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    TFT = (f'TFT & {performance_metrics}')

    start = time.time()
    models = [RNN(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.RNN[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    RNN = (f'RNN & {performance_metrics}')  
    
    start = time.time()
    models = [DilatedRNN(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.DilatedRNN [:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    DilatedRNN = (f'DilatedRNN & {performance_metrics}')  
    
    start = time.time()
    models = [NHITS(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.NHITS[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    NHITS = (f'NHITS & {performance_metrics}')      

    start = time.time()  
    models = [TCN(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.TCN[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    TCN = (f'TCN & {performance_metrics}')
    
    start = time.time()
    models = [BiTCN(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.BiTCN[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    BiTCN = (f'BiTCN & {performance_metrics}')   
    
    start = time.time()  
    models = [LSTM(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.LSTM[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    LSTM = (f'LSTM & {performance_metrics}')    

    start = time.time()  
    models = [NBEATS(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.NBEATS[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    NBEATS = (f'NBEATS & {performance_metrics}')  

    start = time.time()  
    models = [NBEATSx(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.NBEATSx[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    NBEATSx = (f'NBEATSx & {performance_metrics}') 

    start = time.time()  
    models = [GRU(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.GRU[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    GRU = (f'GRU & {performance_metrics}') 

    start = time.time()  
    models = [Informer(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()   
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.Informer[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    Informer = (f'Informer & {performance_metrics}') 
    
    start = time.time()  
    models = [TiDE(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.TiDE[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    TiDE = (f'TiDE & {performance_metrics}') 

    start = time.time()
    models = [PatchTST(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.PatchTST[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    PatchTST = (f'PatchTST & {performance_metrics}') 
    
    start = time.time()
    models = [FEDformer(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.FEDformer[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    FEDformer = (f'FEDformer & {performance_metrics}') 
    
    start = time.time()
    models = [DeepAR(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.DeepAR[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    DeepAR = (f'DeepAR & {performance_metrics}') 
    
    start = time.time()
    models = [TimesNet(input_size=input_size, h=horizon, max_steps=max_steps)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=Y_train_df)
    Y_hat_df = nf.predict().reset_index()
    end = time.time()
    time_s = end - start
    y_pred = Y_hat_df.TimesNet[:horizon]
    performance_metrics = performance(y_true, y_pred, time_s)
    TimesNet = (f'TimesNet & {performance_metrics}') 

    print(MLP)
    print(TiDE)
    print(NHITS)
    print(NBEATS)
    print(NBEATSx)
    
    print(RNN)
    print(DilatedRNN)
    print(LSTM)
    print(GRU)
    print(DeepAR)
    
    print(TFT)
    print(Informer)
    print(PatchTST)
    print(FEDformer)

    print(TCN)
    print(BiTCN)
    print(TimesNet)   

'''
print(MLP)
print(TiDE)
print(NHITS)
print(NBEATS)
print(NBEATSx)

print(RNN)
print(DilatedRNN)
print(LSTM)
print(GRU)
print(DeepAR)

print(TFT)
print(Informer)
print(PatchTST)
print(FEDformer)

print(TCN)
print(BiTCN)
print(TimesNet)  
print(Our)'''
