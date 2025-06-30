import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# user id to recommend restaurants for
USER_ID = "u00000"

# load model from pickle file
with open("./model/model.pkl", "rb") as f:
    model: NearestNeighbors = pickle.load(f)

# load user and restaurant data
user_df = pd.read_parquet("./data/user.small.parquet")
restaurant_df = pd.read_parquet("./data/restaurant.parquet").set_index("index")

# find 20 nearest neighbors to be recommend restaurants
difference, ind = model.kneighbors(
    user_df[user_df["user_id"] == USER_ID].drop(columns="user_id"), n_neighbors=20
)

# get restaurant id from restaurant indices returned from the model
recommend_df = restaurant_df.loc[ind[0]]

# set distance as restaurant score
recommend_df["difference"] = difference[0]

# print the result in json format
print(recommend_df[["restaurant_id", "difference"]].to_json(orient="records", indent=2))
