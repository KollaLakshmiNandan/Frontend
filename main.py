from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from tkinter import ttk

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import math
import time
from numpy import random

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os, joblib

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


global filename
global classifier
global df, sc, scaler
global le

def upload():
    global filename
    global df, le
    filename = filedialog.askopenfilename(initialdir = "Datasets")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    df= pd.read_csv(filename)
    text.insert(END,str(df.head))

def preprocess():
    global df, scaler
    global X, y, X_train, X_test, y_train, y_test, sc, le
    text.delete('1.0', END)
    
    text.insert(END,str(df.head))
    text.insert(END,str(df.describe().T))
    
    df['Date of Launch'] = pd.to_datetime(df['Date of Launch'], dayfirst=True, errors='coerce')
    
    # Extract date components: Day, Month, Year
    df['Day'] = df['Date of Launch'].dt.day
    df['Month'] = df['Date of Launch'].dt.month
    df['Year'] = df['Date of Launch'].dt.year
    df = df.drop(['Date of Launch'], axis = 1)
    label = ['Name of Satellite, Alternate Names',
     'Current Official Name of Satellite',
     'Country/Org of UN Registry',
     'Country of Operator/Owner',
     'Operator/Owner',
     'Users',
     'Purpose',
     'Detailed Purpose',
     'Class of Orbit',
     'Type of Orbit',
     'Contractor',
     'Country of Contractor',
     'Launch Site',
     'Launch Vehicle',
     'COSPAR Number']
    
    le = LabelEncoder()
    for i in label:
        df[i] = le.fit_transform(df[i])
    df.head().T
    
    df['Dry Mass (kg.)'] = df['Dry Mass (kg.)'].fillna(df['Dry Mass (kg.)'].mean())
    df['Power (watts)'] = df['Power (watts)'].fillna(df['Power (watts)'].mean())
    df['Expected Lifetime (yrs.)'] = df['Expected Lifetime (yrs.)'].fillna(df['Expected Lifetime (yrs.)'].mean())
    
    n_samples = 5000
    
    resampled_data = resample(df, replace=True, n_samples=n_samples, random_state=42)
    df = pd.concat([df, resampled_data], ignore_index=True)

    X = df.drop(['Expected Lifetime (yrs.)'], axis = 1)
    y = df['Expected Lifetime (yrs.)']
    print("X.shape:", X.shape)
    print("X:", X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    text.insert(END, "\n\nTotal Records used for training: " + str(len(X_train)) + "\n")
    text.insert(END, "\n\nTotal Records used for testing: " + str(len(X_test)) + "\n")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("X_test.shape:", X_test.shape)
    
    # Checking variable distribution
    for index in range(len(df.columns) - 1):  # Prevent target column normalization
        df.iloc[:, index] = (df.iloc[:, index] - df.iloc[:, index].mean()) / df.iloc[:, index].std()
    df.hist(figsize=(15, 15))


    sns.histplot(y, kde=True, bins=10)
    plt.title('Histplot of Target')

    plt.figure(figsize=(15, 15))
    sns.set(font_scale=1)
    sns.heatmap(df.corr(), cmap='GnBu_r', annot=True, square=True, linewidths=.5)
    plt.title('Variable Correlation')
    plt.show()

    
mae_list = []
mse_list = []
rmse_list = []
r2_list = []

def calculateMetrics(algorithm, predict, testY):
    global X_train, X_test, y_train, y_test 
    # Regression metrics
    mae = mean_absolute_error(testY, predict)
    mse = mean_squared_error(testY, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(testY, predict)
    
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)
    
    print(f"{algorithm} Mean Absolute Error (MAE): {mae:.2f}")
    print(f"{algorithm} Mean Squared Error (MSE): {mse:.2f}")
    print(f"{algorithm} Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"{algorithm} R-squared (R²): {r2:.2f}")
    
    text.insert(END, "Performance Metrics of " + str(algorithm) + "\n")
    text.insert(END, "Mean Absolute Error (MAE): " + str(mae) + "\n")
    text.insert(END, "Mean Squared Error (MSE): " + str(mse) + "\n")
    text.insert(END, "Root Mean Squared Error (RMSE): " + str(rmse) + "\n")
    text.insert(END, "R-squared (R²): " + str(r2) + "\n\n")
    # Convert to pandas Series for better compatibility with seaborn
    testY_series = pd.Series(testY.ravel())  # Ensure it's a 1-D array
    predict_series = pd.Series(predict.ravel())  # Ensure it's a 1-D array

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=testY_series, y=predict_series, alpha=0.6)
    plt.plot([min(testY_series), max(testY_series)], [min(testY_series), max(testY_series)], 'r--', lw=2)  # Line of equality
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(algorithm)
    plt.grid(True)
    plt.show()


def RidgeRegressorModel():
    global ridge, X_train, X_test, y_train, y_test
    global predict

    if os.path.exists('Model/RidgeRegressor.pkl'):
        # Load the trained Ridge model from the file
        ridge = joblib.load('Model/RidgeRegressor.pkl')
        print("Model loaded successfully.")
        predict = ridge.predict(X_test)
        calculateMetrics("Ridge Regressor", predict, y_test)
    else:
        # Train the Ridge model
        ridge = Ridge()  # Customize parameters as needed
        ridge.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(ridge, 'Model/RidgeRegressor.pkl')
        print("Model saved successfully.")
        predict = ridge.predict(X_test)
        calculateMetrics("Ridge Regressor", predict, y_test)


def Linearregression():
    global LR, X_train, X_test, y_train, y_test 
    #text.delete('1.0', END)
    global predict
   
    
    if os.path.exists('Model/LinearRegressor.pkl'):
        # Load the trained model from the file
        LR = joblib.load('Model/LinearRegressor.pkl')
        print("Model loaded successfully.")
        predict = LR.predict(X_test)
        calculateMetrics("Linear Regressor", predict, y_test)
    else:
        # Train the model (assuming X_train and y_train are defined)
        LR=LinearRegression()
        LR.fit(X_train, y_train)
        # Save the trained model to a file
        joblib.dump(LR, 'Model/LinearRegressor.pkl')
        print("Model saved successfully.")
        predict = LR.predict(X_test)
        calculateMetrics("Linear Regressor", predict, y_test)
        

def LSTMModel():
    global rnn_model, X_train, X_test, y_train, y_test
    global predict
    global rnn_model, feature_extractor, X_train, X_test, y_train, y_test
    global predict, extracted_features_train, extracted_features_test
    
    # Reshape your data for RNN input
    # Use timesteps = 1 as there are no time components
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # (samples, timesteps, features)
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  # (samples, timesteps, features)

    if os.path.exists('Model/LSTMModel.h5'):
        from tensorflow.keras.models import load_model
        rnn_model = load_model('Model/LSTMModel.h5')
        print("Model loaded successfully.")
        feature_extractor = Model(inputs=rnn_model.input, outputs=rnn_model.layers[-3].output)  # Output from the second LSTM layer

        # Extract features
        extracted_features_train = feature_extractor.predict(X_train_reshaped)
        extracted_features_test = feature_extractor.predict(X_test_reshaped)

        # You can now calculate metrics for the original RNN predictions if needed
        predict = rnn_model.predict(X_test_reshaped)
        calculateMetrics("LSTM Model", predict, y_test)
    else:
        # Define the RNN model
        rnn_model = Sequential()
        rnn_model.add(LSTM(100, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
        rnn_model.add(Dropout(0.2))
        rnn_model.add(LSTM(80, return_sequences=False))
        rnn_model.add(Dropout(0.2))
        #rnn_model.add(LSTM(50, return_sequences=False))
        #rnn_model.add(Dropout(0.2))
        rnn_model.add(Dense(1))  # Output layer for regression

        # Compile the model
        rnn_model.compile(optimizer='adam', loss='mean_squared_error')

        # Define early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Fit the model
        rnn_model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        # Save the trained model to a file
        rnn_model.save('Model/LSTMModel.h5')
        print("Model saved successfully.")
        feature_extractor = Model(inputs=rnn_model.input, outputs=rnn_model.layers[-3].output)  # Output from the second LSTM layer

        # Extract features
        extracted_features_train = feature_extractor.predict(X_train_reshaped)
        extracted_features_test = feature_extractor.predict(X_test_reshaped)

        # You can now calculate metrics for the original RNN predictions if needed
        predict = rnn_model.predict(X_test_reshaped)
        calculateMetrics("LSTM Model", predict, y_test)

def predict():
    global scaler, rnn_model, le
    # Open file for new data
    file = filedialog.askopenfilename(initialdir="Datasets")
    test = pd.read_csv(file)
    
    # Display selected file info
    text.delete('1.0', END)
    text.insert(END, f'{file} Loaded for Prediction\n\n')
    
    # Ensure consistency with training data preprocessing steps
    test['Date of Launch'] = pd.to_datetime(test['Date of Launch'], errors='coerce')
    
    # Drop rows where 'Date of Launch' could not be converted
    test.dropna(subset=['Date of Launch'], inplace=True)
    
    # Extract date components
    test['Day'] = test['Date of Launch'].dt.day
    test['Month'] = test['Date of Launch'].dt.month
    test['Year'] = test['Date of Launch'].dt.year
    test = test.drop(['Date of Launch'], axis=1)
    
    # Apply Label Encoding using the same encoder as in preprocessing
    categorical_features = [
        'Name of Satellite, Alternate Names',
        'Current Official Name of Satellite',
        'Country/Org of UN Registry',
        'Country of Operator/Owner',
        'Operator/Owner',
        'Users',
        'Purpose',
        'Detailed Purpose',
        'Class of Orbit',
        'Type of Orbit',
        'Contractor',
        'Country of Contractor',
        'Launch Site',
        'Launch Vehicle',
        'COSPAR Number'
    ]
    
    le = LabelEncoder()

    for feature in categorical_features:
        test[feature] = le.fit_transform(test[feature])
    
    # Fill missing values in 'Dry Mass (kg.)' and 'Power (watts)' with mean
    test['Dry Mass (kg.)'] = test['Dry Mass (kg.)'].fillna(test['Dry Mass (kg.)'].mean())
    test['Power (watts)'] = test['Power (watts)'].fillna(test['Power (watts)'].mean())
    
    test_scaled = scaler.transform(test)

    test_scaled_reshaped = test_scaled.reshape((test_scaled.shape[0], 1, test_scaled.shape[1]))  
    
    predictions = rnn_model.predict(test_scaled_reshaped)
    test['satelite life time years']=predictions
    
    text.insert(END, "Predictions:\n" + str(test) + "\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, color='blue', label='Predicted Lifetime')
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Expected Lifetime (yrs.)")
    plt.title("Satellite Lifetime Predictions")
    plt.legend()
    plt.grid(True)
    plt.show()

            
def graph():
    columns = ["Algorithm Name", "MAE", "MSE", "RMSE", "R² Score"]
    algorithm_names = ["Ridge Regression", "Linear Regression", "LSTM Regressor"]
    
    # Combine metrics into a DataFrame
    values = []
    for i in range(len(algorithm_names)):
        values.append([algorithm_names[i], mae_list[i], mse_list[i], rmse_list[i], r2_list[i]])
    
    temp = pd.DataFrame(values, columns=columns)
    text.insert(END, "All Model Performance metrics:\n")
    text.insert(END, str(temp) + "\n")

    metrics = ["MAE", "MSE", "RMSE", "R² Score"]
    index = np.arange(len(algorithm_names))  # Positions of the bars

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2  # Width of the bars
    opacity = 0.8

    # Plotting each metric with an offset
    plt.bar(index, mae_list, bar_width, alpha=opacity, color='b', label='MAE')
    plt.bar(index + bar_width, mse_list, bar_width, alpha=opacity, color='g', label='MSE')
    plt.bar(index + 2 * bar_width, rmse_list, bar_width, alpha=opacity, color='r', label='RMSE')
    plt.bar(index + 3 * bar_width, r2_list, bar_width, alpha=opacity, color='y', label='R² Score')

    # Labeling the chart
    plt.xlabel('Algorithm')
    plt.ylabel('Scores')
    plt.title('Performance Comparison of Models')
    plt.xticks(index + bar_width, algorithm_names)  # Setting the labels for x-axis (algorithms)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def close():
  main.destroy()
   

# Initialize the main window
# Create main window
main = Tk()
main.title("Satellite Expected Lifetime Prediction")
main.state("zoomed")

# Set color scheme
PRIMARY_COLOR = "#1e3d59"  # Dark blue
SECONDARY_COLOR = "#44a9ff"  # Light blue
ACCENT_COLOR = "#f5f0e1"  # Off-white
TEXT_COLOR = "#ffffff"  # White
BACKGROUND_COLOR = "#0a1929"  # Very dark blue

# Create styles
style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', 
                background=PRIMARY_COLOR, 
                foreground=TEXT_COLOR, 
                font=('Segoe UI', 11, 'bold'),
                padding=10,
                relief="flat")
style.map('TButton', 
          background=[('active', SECONDARY_COLOR), ('pressed', PRIMARY_COLOR)],
          foreground=[('active', TEXT_COLOR), ('pressed', TEXT_COLOR)])
# Function to set single background image
def set_background():
    # Create a frame specifically for background
    bg_frame = Frame(main)
    bg_frame.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Path for the background image - replace with your actual path
    bg_image_path = r"C:\Users\IDEAPAD GAMING\Downloads\images.jpeg"
    
    try:
        # Get window dimensions
        width = main.winfo_width() or 1200  # Default if not yet realized
        height = main.winfo_height() or 800  # Default if not yet realized
        
        # Load and resize image to fit window
        img = Image.open(bg_image_path)
        img = img.resize((width, height))
        
        # Darken the image slightly for better contrast with text
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.7)
        
        photo_img = ImageTk.PhotoImage(img)
        
        # Store reference to prevent garbage collection
        main.bg_image = photo_img
        
        # Create label for the image
        img_label = Label(bg_frame, image=photo_img, borderwidth=0)
        img_label.place(x=0, y=0, relwidth=1, relheight=1)
        
    except Exception as e:
        print(f"Error loading background image: {e}")
        # Create solid color background as fallback
        fallback_label = Label(bg_frame, bg=BACKGROUND_COLOR, borderwidth=0)
        fallback_label.place(x=0, y=0, relwidth=1, relheight=1)
    
    return bg_frame

# Function to update background on resize
def update_background(event=None):
    # Only process events from the main window
    if event and event.widget != main:
        return
        
    # Remove old background frame
    if hasattr(main, 'bg_frame'):
        main.bg_frame.destroy()
    
    # Create new background with updated dimensions
    main.bg_frame = set_background()
    
    # Make sure the background is behind all other elements
    main.bg_frame.lower()
    
    # Force update the overlay and content layers to stay on top
    if hasattr(main, 'overlay'):
        main.overlay.lift()
    if hasattr(main, 'content_frame'):
        main.content_frame.lift()

# Create initial background after window is realized
main.update_idletasks()  # Force geometry update
main.bg_frame = set_background()

# Bind resize event with debouncing
resize_after_id = None
def delayed_resize(event):
    global resize_after_id
    if resize_after_id:
        main.after_cancel(resize_after_id)
    resize_after_id = main.after(200, lambda: update_background(event))

main.bind("<Configure>", delayed_resize)

# Semi-transparent overlay for better readability
overlay = Frame(main, bg=BACKGROUND_COLOR)
overlay.place(relx=0.5, rely=0.5, relwidth=0.8, relheight=0.8, anchor=CENTER)
overlay.attributes('-alpha', 0.85) if hasattr(overlay, 'attributes') else None  # Apply transparency if supported
main.overlay = overlay  # Store reference

# Content Frame
content_frame = Frame(main, bg='#131c27')
content_frame.place(relx=0.5, rely=0.5, relwidth=0.78, relheight=0.78, anchor=CENTER)
main.content_frame = content_frame  # Store reference

# Ensure content is on top
overlay.lift()
content_frame.lift()

# Title with gradient effect
title_frame = Frame(content_frame, bg=PRIMARY_COLOR, height=80)
title_frame.pack(fill=X, pady=(0, 10))

title = Label(title_frame, 
              text='AI Tool for Modelling Satellite Expected Lifetime for ISRO Space Missions', 
              justify=CENTER,
              bg=PRIMARY_COLOR,
              fg=TEXT_COLOR,
              font=('Segoe UI', 18, 'bold'),
              height=2)
title.pack(fill=BOTH, expand=True)

# Separator
ttk.Separator(content_frame, orient=HORIZONTAL).pack(fill=X, padx=50, pady=5)

# Button Container - using grid for better organization
button_frame = Frame(content_frame, bg='')
button_frame.pack(fill=BOTH, padx=50, pady=10)

# Configure grid columns to be equally spaced
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)
button_frame.columnconfigure(2, weight=1)

# First row of buttons
upload_btn = ttk.Button(button_frame, text="Upload Satellite Dataset", command=upload)
upload_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

preprocess_btn = ttk.Button(button_frame, text="Data Analysis & Preprocessing", command=preprocess)
preprocess_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

ridge_btn = ttk.Button(button_frame, text="Ridge Regressor", command=RidgeRegressorModel)
ridge_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

# Second row of buttons
linear_btn = ttk.Button(button_frame, text="Linear Regressor", command=Linearregression)
linear_btn.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

lstm_btn = ttk.Button(button_frame, text="LSTM Model", command=LSTMModel)
lstm_btn.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

metrics_btn = ttk.Button(button_frame, text="Performance Metrics", command=graph)
metrics_btn.grid(row=1, column=2, padx=10, pady=10, sticky="ew")

# Third row of buttons
predict_btn = ttk.Button(button_frame, text="Predict on Test Data", command=predict)
predict_btn.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

close_btn = ttk.Button(button_frame, text="Close Application", command=close)
close_btn.grid(row=2, column=2, padx=10, pady=10, sticky="ew")

# Separator before text area
ttk.Separator(content_frame, orient=HORIZONTAL).pack(fill=X, padx=50, pady=5)

# Text area with custom styling
text_frame = Frame(content_frame, bg=ACCENT_COLOR, bd=2, relief=RIDGE)
text_frame.pack(fill=BOTH, expand=True, padx=50, pady=10)
text_frame.pack_propagate(False)  # Prevents the frame from shrinking

# Text widget with improved styling
text = Text(text_frame, 
            bg='#f5f5f5', 
            fg='#333333', 
            font=('Consolas', 11),
            padx=10, 
            pady=10, 
            wrap=WORD,
            borderwidth=0)
text.pack(side=LEFT, fill=BOTH, expand=True)

# Modern scrollbar
scrollbar = ttk.Scrollbar(text_frame, orient=VERTICAL, command=text.yview)
scrollbar.pack(side=RIGHT, fill=Y)
text.configure(yscrollcommand=scrollbar.set)

# Status bar
status_frame = Frame(content_frame, bg=PRIMARY_COLOR, height=25)
status_frame.pack(fill=X, side=BOTTOM)
status_label = Label(status_frame, text="Ready", bg=PRIMARY_COLOR, fg=TEXT_COLOR, anchor=W, padx=10)
status_label.pack(fill=X)

# Function to update status
def update_status(message):
    status_label.config(text=message)

# Start the Tkinter loop
main.mainloop()