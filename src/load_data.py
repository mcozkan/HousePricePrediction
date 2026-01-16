from utils import DataQuality, EdaReports
import pandas as pd
import schemas





# Setting display options for better visibility
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)  

# Load the datasets

df_train = pd.read_csv("/Users/etmco/Downloads/HousePrice/data/train.csv")
df_test = pd.read_csv("/Users/etmco/Downloads/HousePrice/data/test.csv")


# Combine the datasets for unified processing
df_combined = pd.concat([df_train, df_test], axis = 0, sort = False).reset_index(drop = True)

## Pydantic Validation Example
schemas.HousePrice(**df_combined.iloc[0]).dict()