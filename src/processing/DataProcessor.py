import pandas as pd
import src.utils.utilities as utils

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.data = self.discard_columns(self.data)

    def discard_columns(self, data : pd.DataFrame) -> pd.DataFrame:
        """ Discard columns that we will not use 
            TODO: maybe discard "screen_name" and "id"?
        """
        columns = ["_idx", "profile_background_image_url", "profile_image_url"]
        data = data.drop(columns=columns)
        utils.save_tmp_data(data, "discard_columns.csv")
        return data
    
    def normalize_numerical_data(self):
        """ Normalize numerical data using z-score normalization """
        numerical_columns = ["statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count"]
        for column in numerical_columns:
            self.numerical_column_normalization(column)
        utils.save_tmp_data(self.data, "normalize_numerical_data.csv")
        return self.data
    
    def numerical_column_normalization(self, column_name):
        self.data[column_name] = (self.data[column_name] - self.data[column_name].mean()) / self.data[column_name].std()
        return self.data