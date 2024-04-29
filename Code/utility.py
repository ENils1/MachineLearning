
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_error, mean_squared_error, r2_score

def start_xgb(filename):
    df = pd.read_csv(filename, sep=",", encoding="UTF-8")
    ignored_features = ["Assists", "Club Goals", "Goal Difference", "Goals/Game", "Yellow Cards", "Assists/Game", "Red Cards", "Market Value", "Name", "Logarithmic Market Value"]
    X = df.drop(columns=ignored_features)
    y = df['Logarithmic Market Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    return X_train, X_test, y_train, y_test, df

def start(filename):
    df = pd.read_csv(filename, sep=",", encoding="UTF-8")
    X = df.drop(columns=['Logarithmic Market Value', 'Name', 'Market Value'])
    y = df['Logarithmic Market Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    return X_train, X_test, y_train, y_test, df

def train(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the training data
    y_train_pred = model.predict(X_train)
    y_train_original = np.power(10, y_train_pred)
    
    # Calculate MAE, MAPE, R^2, and MSE on the training data
    mae_train = mean_absolute_error(np.power(10, y_train), y_train_original)
    mape_train = np.mean(np.abs((np.power(10, y_train) - y_train_original) / np.power(10, y_train))) * 100
    r2_train = r2_score(np.power(10, y_train), y_train_original)
    mse_train = mean_squared_error(np.power(10, y_train), y_train_original)
    
    print("Training Set Scores:")
    print("Mean Absolute Error (MAE):", round(mae_train / 1000000, 2), "M")
    print("Mean Absolute Percentage Error (MAPE):", round(mape_train, 2), "%")
    print("R-squared (R^2):", round(r2_train, 2))
    print("Mean Squared Error (MSE):", round(mse_train / 1000000, 2), "M")
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_test_original = np.power(10, y_test)
    y_pred_original = np.power(10, y_pred)
    
    # Calculate MAE, MAPE, R^2, and MSE on the test data
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    r2 = r2_score(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    
    print("\nTest Set Scores:")
    print("Mean Absolute Error (MAE):", round(mae / 1000000, 2), "M")
    print("Mean Absolute Percentage Error (MAPE):", round(mape, 2), "%")
    print("R-squared (R^2):", round(r2, 2))
    print("Mean Squared Error (MSE):", round(mse / 1000000, 2), "M")
    
    return y_pred, y_pred_original, y_test_original



def display(df, X_test, y_pred):
    z = pd.merge(X_test, df, on=['Height','Year','Club Goals','Club Goals Conceded','Goals','Assists','Red Cards','Yellow Cards','Minutes Played','Age','Country Value','Position Value','Club Value','League Value'])
    z.set_index(X_test.index, inplace=True)
    z['Logarithmic Predicted Value'] = y_pred
    z['Predicted Value'] = np.power(10, y_pred)
    results = z[['Name', 'Year', 'Market Value', 'Predicted Value']]
    results = results.sort_values(by='Market Value', ascending=False)
    return results, z


def start_legacy(filename):
    df = pd.read_csv(filename, sep=",", encoding="UTF-8")
    X = df.drop(columns=['Logarithmic Market Value'])
    y = df['Logarithmic Market Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    return X_train, X_test, y_train, y_test, df

