import os

import pandas as pd


def get_data(path: str) -> pd.DataFrame:
    output = pd.read_csv(path, encoding="ISO-8859-1")
    print(f"Loading data from {path}; Len: {output.shape[0]}")
    return output


def save_tmp_data(data: pd.DataFrame, name: str):
    """Save data to a temporary file"""
    output_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(output_dir, exist_ok=True)

    data.to_csv(os.path.join(output_dir, name), index=False)


def get_categorical_columns() -> list:
    cat_columns = [
        "account_type",  # 0, 1; is human or bot?
        "url",  # 0, 1; is URL present?
        "location",  # 0, 1; is location present?
        "default_profile",  # 0, 1; is default profile present?
        "default_profile_image",  # 0, 1; is default profile image present?
        "geo_enabled",  # 0, 1; is geo enabled?
        "profile_banner_url",  # 0, 1; is profile banner URL present?
        "profile_use_background_image",  # 0, 1; is profile background image present?
        "description_contins_link",  # 0, 1; does description contain a link?
        "created_time_of_day",
        "created_day_of_week",
    ]
    return cat_columns


def get_date_feature_columns() -> list:
    date_columns = ["created_at"]
    return date_columns


def get_text_feature_columns() -> list:
    text_columns = ["description", "screen_name"]
    return text_columns


def ignore_features_columns() -> list:
    ignore_columns = ["id", "location"]
    return ignore_columns
