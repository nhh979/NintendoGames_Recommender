# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:14:39 2023

@author: hoang
"""

import pandas as pd
import numpy as np

df = pd.read_csv("Datasets/NintendoSwitchGames.csv")

# Title
df['Title'] = df['Title'].str.replace('\n', '')

# Developers
df['Developers'] = df['Developers'].str.replace('\n', '')

# Publisher: Replace missing publisher with the value in developer column
df['Publisher'] = df['Publisher'].fillna(df['Developers']).str.replace('\n', '')

# Genre
df['Genre'] = df['Genre'].str.split(')').str.get(1)
# Remove square brackets and its contents
df['Genre'] = df['Genre'].str.replace('\[[^\]]*\]', '')

# Mode
df['Mode'] = df['Mode'].apply(lambda x: x if pd.isnull(x) 
                             else ('Multiplayer' if 'multi' in x.lower() else 'Single-player'))

# Gameplay
df['Gameplay'] = df['Gameplay'].str.replace('\[.*?\]+', '')
df['Gameplay'] = df['Gameplay'].str.replace('\n', '').str.replace('Gameplay', '').str.replace('Game-play', '')
df['Gameplay'] = df['Gameplay'].apply(lambda x: np.nan if not pd.isna(x) and len(x) < 10 else x)
# Plot
df['Plot'] = df['Plot'].str.replace('\[.*?\]+', '')
df['Plot'] = df['Plot'].str.replace('\n', '').str.replace('Plot', '')
df['Plot'] = df['Plot'].apply(lambda x: np.nan if not pd.isna(x) and len(x) < 10 else x)

# Deal with missing values in the Gameplay and Plot columns
df = df.dropna(how='all', subset=['Gameplay', 'Plot'])
df['Gameplay'] = df['Gameplay'].fillna(df['Plot'])
df['Plot'] = df['Plot'].fillna(df['Gameplay'])


# NaN in Genre and Mode columns
    # Sort the dataset by Developers and Title, then use forward fill method to fill the missing value
df = df.sort_values(['Developers', 'Title'])
df['Genre'] = df['Genre'].fillna(method='ffill')
df['Mode'] = df['Mode'].fillna(method='ffill')
df.sort_index(inplace=True)

# Release Date
df['ReleaseDate'] = df['ReleaseDate'].str.strip().fillna('TBA')


# Save the dataset
df.to_csv("Datasets/cleaned_data.csv", index=False)
