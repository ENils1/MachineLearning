import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def start(filename):
    df = pd.read_csv(f"../DataSets/EncodedData/{filename}", sep=",", encoding="UTF-8")
    X = df.drop(columns=['Logarithmic Market Value', 'Name', 'Market Value', "sub_position"])
    y = df['Logarithmic Market Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    return X_train, X_test, y_train, y_test, df

def train(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the training data
    y_train_pred = model.predict(X_train)
    y_train_original = np.power(10, y_train_pred)
    
    # Calculate MAE and MAPE on the training data
    mae_train = mean_absolute_error(np.power(10, y_train), y_train_original)
    mape_train = np.mean(np.abs((np.power(10, y_train) - y_train_original) / np.power(10, y_train))) * 100
    
    print("Training Set Scores:")
    print("Mean Absolute Error (MAE):", round(mae_train / 1000000, 2), "M")
    print("Mean Absolute Percentage Error (MAPE):", round(mape_train, 2), "%")
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_test_original = np.power(10, y_test)
    y_pred_original = np.power(10, y_pred)
    
    # Calculate MAE and MAPE on the test data
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    
    print("\nTest Set Scores:")
    print("Mean Absolute Error (MAE):", round(mae / 1000000, 2), "M")
    print("Mean Absolute Percentage Error (MAPE):", round(mape, 2), "%")

    return y_pred, y_pred_original, y_test_original


def display(df, X_test, y_pred):
    z = pd.merge(X_test, df, on=['height_in_cm','year','goals_for','goals_against','goals','assists','red_cards','yellow_cards','minutes_played','age_at_evaluation','country_of_citizenship_encoded','sub_position_encoded','club_id_encoded','domestic_competition_id_encoded'])
    z.set_index(X_test.index, inplace=True)
    z['Predicted_Value_log'] = y_pred
    z['Predicted_Value'] = np.power(10, y_pred)
    results = z[['name', 'year', 'market_value_in_eur', 'Predicted_Value']]
    results = results.sort_values(by='market_value_in_eur', ascending=False)
    return results, z

