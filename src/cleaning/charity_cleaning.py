## Script to clean raw charity data based on EDA. EDA can be found in eda_charity notebook
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np


# Load Data
print('Loading charity data')
root_dir =  os.path.abspath(os.path.join(__file__ ,"../../.."))
file_name = os.path.join(root_dir, "data","external", "charity_data_raw.csv")
all_charity = pd.read_csv(file_name)

# Filter out charities based on the following qualities
# 1. Have a charity rating score within the bottom 25% of scores
# 2. Have a mission statement with fewer than 10 words
# 3. Are religious organizations or a United Ways (each local united way has its own listing and adds noise to the reccommender)

# Drop charities with scores within bottom 25%
print('Dropping charities with low ratings')
all_charity_use = all_charity[all_charity['score']>np.percentile(all_charity['score'], 25)]

# Drop charities with mission statements with fewer than 10 words
print('Dropping charities with few words in mission statement')
all_charity_use['description_length'] = all_charity_use.apply(lambda x: len(x['description'].split(" ")),axis=1)
all_charity_use = all_charity_use[all_charity_use['description_length']>10]

# Drop religious charities and united ways
print('Dropping religious charities and united ways')
all_charity_use = all_charity_use[all_charity_use['category'] != 'Religion']
all_charity_use = all_charity_use[~all_charity_use['subcategory'].isin(['United Ways', 'Jewish Federations'])]

# Remove charity name from description
all_charity_use['description_noname'] = all_charity_use.apply(lambda x: x['description'].replace(x['name'],""),axis=1)

# Trim dataset to only columns we need
all_charity_use = all_charity_use[['name', 'score','description','description_noname','category','subcategory']]

# Save cleaned dataset
print('Saving cleaned charity data')
file_save = os.path.join(root_dir, "data","processed", "charity_data_cleaned.csv")
all_charity_use.to_csv(file_save)
