import pandas as pd
import src.utils.utilities as utils
import seaborn as sns

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.data = self.discard_columns(self.data)
        self.data = self.replace_missing_values(self.data, "lang", "n/a")
        self.data = self.convert_date_to_datetime64(self.data)
        self.data = self.add_time_of_day(self.data)
        self.data = self.add_day_of_week(self.data)
        self.data = self.convert_account_type_to_int(self.data)

    def discard_columns(self, data : pd.DataFrame) -> pd.DataFrame:
        """ Discard columns that we will not use 
            TODO: maybe discard "screen_name" and "id"?
        """
        columns = ["_idx", "id", "profile_background_image_url", "profile_image_url"]
        data = data.drop(columns=columns)
        utils.save_tmp_data(data, "discard_columns.csv")
        return data
    
    def replace_missing_values(self, data : pd.DataFrame, column, value) -> pd.DataFrame:
        """ Replace missing values with the mean of the column """
        data[column] = data[column].fillna(value)
        utils.save_tmp_data(data, "replace_missing_values.csv")
        return data
    
    def convert_date_to_datetime64(self, data : pd.DataFrame) -> pd.DataFrame:
        """ Convert date to datetime64. Necessary for Pycaret. """
        data['created_at'] = pd.to_datetime(data['created_at'], format='%Y-%m-%d %H:%M:%S')
    
        utils.save_tmp_data(data, "convert_date_to_datetime64.csv")
        return data
    
    def add_time_of_day(self, data : pd.DataFrame) -> pd.DataFrame:
        """ Add time of day as 1 to 24 as a feature """
        data['created_time_of_day'] = data['created_at'].dt.hour
        utils.save_tmp_data(data, "add_time_of_day.csv")
        return data
    
    def add_day_of_week(self, data : pd.DataFrame) -> pd.DataFrame:
        """ Add day of week as 1 to 7 as a feature """
        data['created_day_of_week'] = data['created_at'].dt.dayofweek
        utils.save_tmp_data(data, "add_day_of_week.csv")
        return data
    
    def convert_account_type_to_int(self, data : pd.DataFrame) -> pd.DataFrame:
        """ Convert account type to int """
        data['account_type_int'] = data['account_type'].map({'human': 0, 'bot': 1})
        utils.save_tmp_data(data, "convert_account_type_to_int.csv")
        return data