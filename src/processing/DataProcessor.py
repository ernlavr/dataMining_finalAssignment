import os
import pandas as pd
import src.utils.utilities as utils
import seaborn as sns
import locale
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

class DataProcessor:
    def __init__(self, args):
        self.user_file = "users.csv"
        self.tweet_file = "tweets.csv"

        self.data_human = self.process_dataset(args.humans_folder, 1)
        self.data_bot = self.process_dataset(args.bots_folder, 0)

        # assert if data_human and data_bot have identical columns
        assert self.data_human.columns.equals(self.data_bot.columns)

        self.data_merged = pd.concat([self.data_human, self.data_bot])
        self.data_merged = self.process_merged(self.data_merged)

        # print length of data
        print(f"Length of data: {len(self.data_merged)}")
        # remove all non numeric columns
        numeric_df = self.data_merged.select_dtypes(include='number')
        # remove outliers
        self.data_merged = numeric_df[(np.abs(stats.zscore(numeric_df)) < 3).all(axis=1)]
        print(f"Length of data after removing outliers: {len(self.data_merged)}")

        # Run a few initial visualization
        self.visualize(self.data_merged)

    def visualize(self, data : pd.DataFrame):
        # visualize the correlation
        corr = data.corr()
        f, ax = plt.subplots(figsize=(14, 10))
        hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
        t= f.suptitle('Twitter Bot Account Feature Correlation Heatmap', fontsize=14)
        
        # Data folder
        output_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_correlation_no_outliers.png"))

    def process_merged(self, data : pd.DataFrame) -> pd.DataFrame:
        # create lang categories
        data["lang_cat"] = data["lang"].astype("category").cat.codes
        
        # save
        utils.save_tmp_data(data, "process_merged.csv")
        return data

    def process_dataset(self, datapath, human : int):
        df = utils.get_data(os.path.join(datapath, self.user_file))

        # add human/bot label
        df["account_type"] = human

        # convert date to datetime64
        df = self.convert_date_to_datetime64(df)

        # add time of day as 1 to 24 as a feature
        df = self.add_time_of_day(df)

        # add day of week as 1 to 7 as a feature
        df = self.add_day_of_week(df)

        # is "url" present?
        df["url"] = df["url"].notnull().astype(int)

        # is "location" present?
        df["location"] = df["location"].notnull().astype(int)

        # is "default_profile" present? Add as 0 or 1
        df["default_profile"] = df["default_profile"].notnull().astype(int)

        # is "default_profile_image" present?
        df["default_profile_image"] = df["default_profile_image"].notnull().astype(int)

        # is "geo_enabled" present?
        df["geo_enabled"] = df["geo_enabled"].notnull().astype(int)

        # drop "profile_image_url"
        df = df.drop(columns=["profile_image_url"])

        # is "profile_banner_url" not "NULL"?
        df["profile_banner_url"] = df["profile_banner_url"].notnull().astype(int)

        # is "profile_use_background_image" present?
        df["profile_use_background_image"] = df["profile_use_background_image"].notnull().astype(int)

        # drop columns because they are not useful
        columns_to_drop = [
            "profile_background_image_url_https",
            "profile_text_color",
            "profile_image_url_https",
            "profile_sidebar_border_color",
            "profile_background_tile",
            "profile_sidebar_fill_color",
            "profile_background_image_url",
            "profile_background_color",
            "profile_link_color",
            "utc_offset",
            "protected",
            "verified"
        ]

        # drop all columns at once
        df = df.drop(columns=columns_to_drop)

        # add "." to descriptions which are empty
        df["description"] = df["description"].fillna(".")

        # "description" number of characters
        df["description_num_char"] = df["description"].str.len()

        # "description" contains http:// or https:// ?
        df["description_contins_link"] = df["description"].str.contains("http://|https://")

        df = self.parse_tweets(df, datapath)

        # drop "created_at" because we have "account_age_days"
        df = df.drop(columns=["created_at"])

        # save tmp data
        data_name = "human" if human == 1 else "bot"
        utils.save_tmp_data(df, f"process_dataset_{data_name}.csv")

        return df

    def parse_tweets(self, users_df, datapath):
        """ For both users.csv files, map the user IDs to the tweets.csv 
        and extract useful information such as

        - average number of retwets
        - average number of favorites
        - mean num_urls
        - mean num_mentions
        - mean num_hashtags
        - average length of tweet

        - median day of tweeting
        - median time of tweeting
        - sentiment of tweet?
        """
        tweets_path = os.path.join(datapath, self.tweet_file)
        tweets_df = utils.get_data(tweets_path)

        def parse_by_user_id(self, user_id, tweets_df):
            """ Extract all tweets by user_id 
                # TODO: outline in the report that means are sensetive to outliers
                e.g. user ID 16119337
            """
            tweets = tweets_df[tweets_df["user_id"] == user_id]

            # average number of retwets, if null, return 0
            avg_retweets = tweets["retweet_count"].mean()
            avg_retweets = avg_retweets if pd.notna(avg_retweets) else 0

            # average number of favorites
            avg_favorites = tweets["favorite_count"].mean()
            avg_favorites = avg_favorites if pd.notna(avg_favorites) else 0

            # average length of tweet
            avg_length = tweets["text"].str.len().mean()
            avg_length = avg_length if pd.notna(avg_length) else 0

            # convert created_at to datetime64
            tweets = self.convert_date_to_datetime64(tweets, save=False)

            # add time of day as 1 to 24 as a feature
            tweets = self.add_time_of_day(tweets, save=False)

            # add day of week as 1 to 7 as a feature
            tweets = self.add_day_of_week(tweets, save=False)

            # extract median day of tweeting
            median_day_of_week = tweets["created_day_of_week"].median()
            median_day_of_week = median_day_of_week if pd.notna(median_day_of_week) else -1

            # extract median time of tweeting
            median_time_of_day = tweets["created_time_of_day"].median()
            median_time_of_day = median_time_of_day if pd.notna(median_time_of_day) else -1

            return avg_retweets, avg_favorites, avg_length, median_day_of_week, median_time_of_day

        users_df[['avg_retweets', 
            'avg_favorites', 
            'avg_length',
            'median_day_of_tweeting',
            'median_time_of_tweeting']] = users_df['id'].apply(lambda x: pd.Series(parse_by_user_id(self, x, tweets_df)))

        # save df
        utils.save_tmp_data(users_df, "parse_tweets.csv")

        return users_df


        
        



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
    
    def convert_date_to_datetime64(self, data : pd.DataFrame, save=True) -> pd.DataFrame:
        """ Convert date to datetime64. Necessary for Pycaret. """
        locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
        data['created_at'] = pd.to_datetime(data['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    
        if save:
            utils.save_tmp_data(data, "convert_date_to_datetime64.csv")
        return data
    
    def add_time_of_day(self, data : pd.DataFrame, save=True) -> pd.DataFrame:
        """ Add time of day as 1 to 24 as a feature """
        data['created_time_of_day'] = data['created_at'].dt.hour
        
        if save:
            utils.save_tmp_data(data, "add_time_of_day.csv")
        return data
    
    def add_day_of_week(self, data : pd.DataFrame, save=True) -> pd.DataFrame:
        """ Add day of week as 1 to 7 as a feature """
        data['created_day_of_week'] = data['created_at'].dt.dayofweek + 1
        
        if save:
            utils.save_tmp_data(data, "add_day_of_week.csv")
        return data
    
    def convert_account_type_to_int(self, data : pd.DataFrame) -> pd.DataFrame:
        """ Convert account type to int """
        data['account_type_int'] = data['account_type'].map({'human': 0, 'bot': 1})
        utils.save_tmp_data(data, "convert_account_type_to_int.csv")
        return data