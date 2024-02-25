
import pandas as pd
import numpy as np


pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 1200)
pd.set_option('display.width', 1200)

df = pd.read_csv("cleaned_data_25.csv", sep=",", encoding="UTF-8")
print(df)

df_sorted = df.sort_values(by=['player_id', 'date'], ascending=[True, False])

result_df = df_sorted.drop_duplicates(subset=['player_id', 'year'], keep='first')

print(result_df)

result_df = result_df.dropna()

result_df.to_csv("cleaned_data_26.csv", index=False)

print(result_df)
