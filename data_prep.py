import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import sys

base_folder = "../dataset/"
output_folder = "dataset/"


def process_gowalla_dataset():
    filename = "Gowalla_totalCheckins.txt"
    datapath = base_folder + filename

    # Read the Gowalla dataset text file with tab delimiter and no header
    data = pd.read_csv(datapath, delimiter="\t", header=None,
                       names=["user_id", "timestamp", "latitude", "longitude", "poi_id"])

    # Convert timestamp to datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Process latitude and longitude
    data["latitude"] = data["latitude"].astype(float)
    data["longitude"] = data["longitude"].astype(float)

    # Map user_id and poi_id to a continuous range of integers
    unique_users = data['user_id'].unique()
    user_to_index = {user: idx for idx, user in enumerate(unique_users)}
    data['user_id'] = data['user_id'].map(user_to_index)

    unique_pois = data['poi_id'].unique()
    poi_to_index = {poi: idx for idx, poi in enumerate(unique_pois)}
    data['poi_id'] = data['poi_id'].map(poi_to_index)

    # Remove users with only one check-in
    user_checkin_count = data.groupby("user_id").size()
    users_to_keep = user_checkin_count[user_checkin_count > 1].index
    data = data[data["user_id"].isin(users_to_keep)]

    # Create a new column "trajectory_id" by combining user_id and check-in sequence
    data["checkin_seq"] = data.groupby("user_id").cumcount() + 1
    data["trajectory_id"] = data["user_id"].astype(
        str) + "_" + data["checkin_seq"].astype(str)

    # Compute additional features
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["norm_in_day_time"] = (data["timestamp"].dt.hour * 3600 +
                                data["timestamp"].dt.minute * 60 + data["timestamp"].dt.second) / 86400
    data["norm_day_shift"] = data.groupby(
        "user_id")["timestamp"].diff().dt.total_seconds() / 86400
    data["norm_relative_time"] = data.groupby(
        "user_id")["norm_in_day_time"].diff()

    # Fill NaN values with 0 for norm_day_shift and norm_relative_time
    data["norm_day_shift"].fillna(0, inplace=True)
    data["norm_relative_time"].fillna(0, inplace=True)

    # Split the data into features (X) and target variable (y)
    X = data[["user_id", "poi_id", "latitude", "longitude", "trajectory_id",
              "day_of_week", "norm_in_day_time", "norm_day_shift", "norm_relative_time"]]

    # Split the data into training and testing sets
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

    # Save the preprocessed data to a pickle file
    print("Gowalla: Saving preprocessed data to pickle file")
    with open(output_folder+"gowalla.pkl", "wb") as file:
        pickle.dump((X_train, X_val, X_test), file)


def process_nyc_dataset():
    filename = "dataset_TSMC2014_NYC.txt"
    datapath = base_folder + filename

    # Read the NYC dataset text file with tab delimiter and no header
    data = pd.read_csv(datapath, delimiter="\t", header=None,
                       names=["user_id", "poi_id", "category_id", "category_name",
                              "latitude", "longitude", "timezone_offset", "timestamp"], encoding='latin1')

    # Convert timestamp to datetime format
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], format="%a %b %d %H:%M:%S %z %Y")

    # Process latitude and longitude
    data["latitude"] = data["latitude"].astype(float)
    data["longitude"] = data["longitude"].astype(float)

    # Map user_id and poi_id to a continuous range of integers
    unique_users = data['user_id'].unique()
    user_to_index = {user: idx for idx, user in enumerate(unique_users)}
    data['user_id'] = data['user_id'].map(user_to_index)

    unique_pois = data['poi_id'].unique()
    poi_to_index = {poi: idx for idx, poi in enumerate(unique_pois)}
    data['poi_id'] = data['poi_id'].map(poi_to_index)

    # Remove users with only one check-in
    user_checkin_count = data.groupby("user_id").size()
    users_to_keep = user_checkin_count[user_checkin_count > 1].index
    data = data[data["user_id"].isin(users_to_keep)]

    # Create a new column "trajectory_id" by combining user_id and check-in sequence
    data["checkin_seq"] = data.groupby("user_id").cumcount() + 1
    data["trajectory_id"] = data["user_id"].astype(
        str) + "_" + data["checkin_seq"].astype(str)

    # Compute additional features
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["norm_in_day_time"] = (data["timestamp"].dt.hour * 3600 +
                                data["timestamp"].dt.minute * 60 + data["timestamp"].dt.second) / 86400
    data["norm_day_shift"] = data.groupby(
        "user_id")["timestamp"].diff().dt.total_seconds() / 86400
    data["norm_relative_time"] = data.groupby(
        "user_id")["norm_in_day_time"].diff()

    # Fill NaN values with 0 for norm_day_shift and norm_relative_time
    data["norm_day_shift"].fillna(0, inplace=True)
    data["norm_relative_time"].fillna(0, inplace=True)

    # Split the data into features (X) and target variable (y)
    X = data[["user_id", "poi_id", "latitude", "longitude", "trajectory_id",
              "day_of_week", "norm_in_day_time", "norm_day_shift", "norm_relative_time"]]

    # Split the data into training and testing sets
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

    # Save the preprocessed data to a pickle file
    print("NYC: Saving preprocessed data to pickle file")
    with open(output_folder+"nyc.pkl", "wb") as file:
        pickle.dump((X_train, X_val, X_test), file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_prep.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]

    if dataset == "gowalla":
        process_gowalla_dataset()
    elif dataset == "nyc":
        process_nyc_dataset()
    else:
        print("Invalid dataset. Please choose 'gowalla' or 'nyc'.")
        sys.exit(1)
