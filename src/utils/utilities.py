import pandas as pd
import argparse
import omegaconf
import os

def get_data(path : str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="ISO-8859-1")

def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--args_path", type=str, default=None, help="Path to args file")
    # humans_folder
    parser.add_argument("--humans_folder", type=str, default="data/e13", help="Path to humans folder")
    parser.add_argument("--bots_folder", type=str, default="data/twt", help="Path to bots folder")

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
        "favourites_count",
        "followers_count",
        "friends_count",
        "statuses_count",
        "average_tweets_per_day",
        "account_age_days"
    ]
    return numeric_columns

def get_categorical_columns() -> list:
    cat_columns = [
        "default_profile",
        "default_profile_image",
        "geo_enabled",
        "lang",
        "verified",
        "account_type",
        "created_day_of_week",
        "created_time_of_day"
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