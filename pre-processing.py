import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import calendar


def read_file(filename):
    df = pd.read_csv(filename, sep=",", encoding="UTF-8")
    print(df.shape)


read_file("RawData/appearances.csv")
read_file("RawData/players.csv")
