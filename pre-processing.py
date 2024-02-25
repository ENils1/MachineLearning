# Import modules

import pandas as pd
import numpy as np
#import warnings
import os

#settings
pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 1200)
pd.set_option('display.width', 1200)
#warnings.filterwarnings("ignore")
#%matplotlib inline

#variables
#colour=['maroon','r','g','darkgreen','c','teal','b','navy','indigo','m','deeppink','orange','sienna','yellow','khaki','olive','tan','black','grey','brown']

print("Packages installed")

# import all files in Data folder and read into dataframes
dataframes=[]
for dirname, _, filenames in os.walk('RawData/'):
    for filename in filenames:
        file=filename.split('.')
        file=((file[0]+"_df"))
        if file !="_df":
            filepath=os.path.join(dirname,filename)
            df=pd.read_csv(filepath,sep=",",encoding = "UTF-8")
            exec(f'{file} = df.copy()')
            print(file, df.shape)
            dataframes.append(df)
print('Data imported') #Tar ca. 10 sekunder å lese alle filene

#0.25 vårsesongen, 0.75 høstsesongen.
def date_to_year(df):
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

    df["year"] = np.where(df["month"] <= 6, df["year"] + 0.25, df["year"] + 0.75)
    return df

def valuation_date_to_year(df):
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

    conditions = [
    (df['month'] > 2) & (df['month'] < 9),
    (df['month'] >= 9),
    (df['month'] < 3)
    ]

    choices = ['Spring', 'Fall', 'Fall']
    df['semester'] = np.select(conditions, choices)
    df.loc[(df['month'] < 3), 'year'] -= 1
    df["year"] = np.where(df["semester"] == "Spring", df["year"] + 0.25, df["year"] + 0.75)
    
    return df

def calculate_age(row):
    date_of_birth = pd.to_datetime(row['date_of_birth'])
    date = pd.to_datetime(row['date'])
    age = date.year - date_of_birth.year - ((date.month, date.day) < (date.month, date.day))
    return age

games_df = games_df[["game_id", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]]
appearances_df = appearances_df[["game_id", "player_id", "player_club_id", "yellow_cards", "red_cards", "goals", "assists", "minutes_played"]]

games_df = pd.merge(games_df, appearances_df, on="game_id")

games_df['goals_for'] = games_df.apply(lambda row: row['home_club_goals'] if row['home_club_id'] == row['player_club_id'] else row['away_club_goals'], axis=1)
games_df['goals_against'] = games_df.apply(lambda row: row['away_club_goals'] if row['home_club_id'] == row['player_club_id'] else row['home_club_goals'], axis=1)
games_df = date_to_year(games_df)
print(games_df)

games_df = games_df[["player_id", "player_club_id", "yellow_cards", "red_cards", "goals", "assists", "minutes_played", "goals_for", "goals_against", "year"]]
player_performance_df = games_df.groupby(['player_id', 'player_club_id', 'year']).agg({
    'goals_for': 'sum',
    'goals_against': 'sum',
    'goals': 'sum', 
    'assists': 'sum', 
    'red_cards': 'sum', 
    'yellow_cards': 'sum', 
    'minutes_played': 'sum'
}).reset_index()

print(player_performance_df)

player_valuations_df = valuation_date_to_year(player_valuations_df)[["player_id", "market_value_in_eur", "year", "date"]]
print(player_valuations_df)

players_performance_value_df = pd.merge(player_performance_df, player_valuations_df, on=["player_id", "year"])
print(player_performance_df)

players_characteristics_df = players_df[['player_id', 'country_of_birth', 'country_of_citizenship','date_of_birth', 'height_in_cm', 'sub_position']]
players_characteristics_df['country_of_birth'].fillna(players_characteristics_df['country_of_citizenship'], inplace=True)
players_characteristics_df.dropna(inplace=True)
total_df = pd.merge(players_characteristics_df, players_performance_value_df, on='player_id', how='left')
total_df.dropna(inplace=True)

total_df['age_at_evaluation'] = total_df.apply(calculate_age, axis=1)


print(total_df)

total_df.to_csv("cleaned_data_25.csv", index=False)

print("donso")