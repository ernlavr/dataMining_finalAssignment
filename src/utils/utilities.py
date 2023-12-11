import pandas as pd
import argparse
import omegaconf
import os

def get_data(path : str) -> pd.DataFrame:
    output = pd.read_csv(path, encoding="ISO-8859-1")
    print(f"Loading data from {path}; Len: {output.shape[0]}")
    return output

def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--args_path", type=str, default=None, help="Path to args file")
    # Data
    parser.add_argument("--humans_folder", type=str, default="data/e13", help="Path to humans folder")
    parser.add_argument("--bots_folder", type=str, default="data/twt", help="Path to bots folder")
    parser.add_argument("--data_train", type=str, default="data/processed/train.csv", help="Path to data_train")
    parser.add_argument("--data_test", type=str, default="data/processed/test.csv", help="Path to data_test")

    # Preprocessing
    parser.add_argument("--preprocess", type=bool, default=False, help="Preprocess data")

    # Clustering
    parser.add_argument("--cluster", type=bool, default=False, help="Cluster data")

    # check if we have a config file
    args = parser.parse_args()
    if args.args_path is not None:
        # load the config file
        args = omegaconf.OmegaConf.load(args.args_path)
        args = omegaconf.OmegaConf.to_container(args, resolve=True)
        
        # convert to args namespace
        args = argparse.Namespace(**args)

    return args

def save_tmp_data(data : pd.DataFrame, name : str):
    """ Save data to a temporary file """
    output_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(output_dir, exist_ok=True)

    data.to_csv(os.path.join(output_dir, name), index=False)

def get_numeric_columns() -> list:
    """ Get the numeric columns of the dataset """
    numeric_columns = [
        "statuses_count",
        "followers_count",
        "friends_count",
        "favourites_count",
        "listed_count",
        "description_num_char",
        "avg_retweets",
        "avg_favorites",
        "avg_length",
        "median_day_of_tweeting", # -1 if no tweets
        "median_time_of_tweeting", # -1 if no tweets
    ]
    return numeric_columns

def get_categorical_columns() -> list:
    cat_columns = [
        "account_type", # 0, 1; is human or bot?
        "url",  # 0, 1; is URL present?
        "location", # 0, 1; is location present?
        "default_profile", # 0, 1; is default profile present?
        "default_profile_image", # 0, 1; is default profile image present?
        "geo_enabled", # 0, 1; is geo enabled?
        "profile_banner_url", # 0, 1; is profile banner URL present?
        "profile_use_background_image", # 0, 1; is profile background image present?
        "description_contins_link", # 0, 1; does description contain a link?
        "created_time_of_day",
        "created_day_of_week"
    ]
    return cat_columns

def get_date_feature_columns() -> list:
    date_columns = [
        "created_at"
    ]
    return date_columns

def get_text_feature_columns() -> list:
    text_columns = [
        "description",
        "screen_name"
    ]
    return text_columns

def ignore_features_columns() -> list:
    ignore_columns = [
        "id",
        "location"
    ]
    return ignore_columns