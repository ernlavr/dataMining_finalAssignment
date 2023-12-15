import os
import pandas as pd
import src.utils.utilities as utils
import seaborn as sns
import locale
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import requests

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class DataProcessor:
    def __init__(self, args):
        self.user_file = "users.csv"
        self.tweet_file = "tweets.csv"

        # Set features
        self.data_human = self.process_dataset(args.humans_folder, 1)
        self.data_bot = self.process_dataset(args.bots_folder, 0)

        # Assert identical lengths and merge
        assert self.data_human.columns.equals(self.data_bot.columns)
        self.data_merged = pd.concat([self.data_human, self.data_bot])
        self.data_merged = self.process_merged(self.data_merged)

        # Remove non-numeric columns and visualize
        numeric_df = self.data_merged.select_dtypes(include='number')
        self.visualize(numeric_df, "feature_correlation")
        
        # Remove outliers and apply min-max scaler
        self.data_merged = self.remove_outliers_by_numerical_cols(numeric_df)
        self.data_merged = self.apply_min_max_scaler(self.data_merged)

        # Plot correlation again
        self.visualize(self.data_merged, "feature_correlation_no_outliers")

        train, test = self.get_train_test()

        # save data
        save_dir = os.path.join(os.getcwd(), "data", "processed")
        os.makedirs(save_dir, exist_ok=True)
        train.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(save_dir, "test.csv"), index=False)
        self.check_duplicates(self.data_merged)
        self.data_merged.to_csv(os.path.join(save_dir, "data_parsed.csv"), index=False) # unsplit for debugging


    def check_duplicates(self, df : pd.DataFrame):
        """ Check if there are any duplicates """
        print(df.duplicated().sum())

    def get_train_test(self):
        """ Balance the dataset by returning a train and test set 
            Train test size should be 80% of the minority class and majority
            class should be downsampled. We're balancing according to "account_type" attribute
        """
        # Separate minority and majority classes
        human_class = self.data_merged[self.data_merged['account_type'] == 1]
        bot_class = self.data_merged[self.data_merged['account_type'] == 0]

        minority_class = human_class if len(human_class) < len(bot_class) else bot_class
        majority_class = human_class if len(human_class) > len(bot_class) else bot_class

        # Set the train size to be 80% of the minority class
        train_size = int(0.8 * len(minority_class))

        # shuffle-up both classes
        minority_class = minority_class.sample(frac=1, random_state=42).reset_index(drop=True)
        majority_class = majority_class.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split the minority class by train_size and leftover
        minority_train = minority_class[:train_size]
        minority_test = minority_class[train_size:]

        # Split the majority class by train_size and leftover
        majority_train = majority_class[:train_size]
        majority_test = majority_class[train_size:]

        # Merge both classes
        train_set = pd.concat([minority_train, majority_train])
        test_set = pd.concat([minority_test, majority_test])

        # test if train_set has equal distribution of human/bot
        print("Train set distribution:")
        print(train_set["account_type"].value_counts())

        # test if test_set has equal distribution of human/bot
        print("Test set distribution:")
        print(test_set["account_type"].value_counts())

        return train_set, test_set



    def visualize(self, data : pd.DataFrame, filename):
        # visualize the correlation
        corr = data.corr()
        f, ax = plt.subplots(figsize=(20, 16))
        hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
        t= f.suptitle('Twitter Bot Account Feature Correlation Heatmap', fontsize=14)
        
        # Data folder
        output_dir = os.path.join(os.getcwd(), "output", "images", __class__.__name__)
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.clf()
        plt.close()


    def remove_outliers_by_numerical_cols(self, data : pd.DataFrame) -> pd.DataFrame:
        """ Remove outliers by columns which are floats """
        numeric_columns = utils.get_numeric_columns()

        # Remove outliers using z-score
        for column in numeric_columns:
            z_scores = stats.zscore(data[column])
            outliers = (z_scores > 3) | (z_scores < -3)
            data = data[~outliers]

        return data
    
    def apply_min_max_scaler(self, data : pd.DataFrame) -> pd.DataFrame:
        """ Apply min max scaler to all numeric columns """
        numeric_cols = utils.get_numeric_columns()
        scaler = MinMaxScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        return data

    def process_merged(self, data : pd.DataFrame) -> pd.DataFrame:
        # create lang categories
        data["lang_cat"] = data["lang"].astype("category").cat.codes
        # one-hot
        encoded = pd.get_dummies(data["lang_cat"], prefix="lang")
        data = pd.concat([data, encoded], axis=1)


        # Remove ID
        data = data.drop(columns=["id"])
        data = data.drop(columns=["lang_cat"])
        
        # save
        utils.save_tmp_data(data, "process_merged.csv")
        return data

    def process_dataset(self, datapath, human : int):
        df = utils.get_data(os.path.join(datapath, self.user_file))

        # add human/bot label
        df["account_type"] = human

        # convert date to datetime64
        df = self.convert_date_to_datetime64(df)

        # one-hot encode "url"
        df = self.one_hot_column(df, "url")

        # is "location" present?
        df = self.one_hot_column(df, "location")

        # is "default_profile" present? Add as 0 or 1
        df = self.one_hot_column(df, "default_profile")

        # is "geo_enabled" present?
        df = self.one_hot_column(df, "geo_enabled")

        # drop "profile_image_url"
        df = df.drop(columns=["profile_image_url"])

        # is "profile_banner_url" not "NULL"?
        df = self.one_hot_column(df, "profile_banner_url")

        # is "profile_use_background_image" present?
        df = self.one_hot_column(df, "profile_use_background_image")

        # drop columns because they are not useful
        columns_to_drop = [
            # Text and styling
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
            
            # Empty columns
            "protected",
            "verified",

            # Low correlation
            "created_at",
            "default_profile_image"
        ]

        # drop all columns at once
        df = df.drop(columns=columns_to_drop)

        # add "." to descriptions which are empty
        df["description"] = df["description"].fillna(".")

        # "description" number of characters
        df["description_num_char"] = df["description"].str.len()

        # "description" contains http:// or https:// ?
        df["description"] = df["description"].str.contains("http://|https://")
        df_encoded = pd.get_dummies(df["description"], prefix="description_contains_link")
        df = pd.concat([df, df_encoded], axis=1)

        df = self.parse_tweets(df, datapath)


        # save tmp data
        data_name = "human" if human == 1 else "bot"
        utils.save_tmp_data(df, f"process_dataset_{data_name}.csv")

        return df

    def one_hot_column(self, data : pd.DataFrame, column : str) -> pd.DataFrame:
        """ One-hot encode a column """
        # one-hot encode "url"
        df_encoded = pd.get_dummies(data[column].notnull().astype(int), prefix=f'{column}_present')
        data = pd.concat([data, df_encoded], axis=1)
        # remove the original column
        data = data.drop(columns=[column])

        return data

    def extract_median_tweet_sentiment(self, tweets):
        # Sentiment analysis
        model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        hf_token = "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" 
        API_URL = "https://api-inference.huggingface.co/models/" + model
        headers = {"Authorization": "Bearer %s" % (hf_token)}
        tweets_analysis = []

        def analysis(tweet):
            payload = dict(inputs=tweet, options=dict(wait_for_model=True))
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        for tweet in tqdm(tweets, "Tweet sentiments"):
            try:
                sentiment_result = analysis(tweet)[0]
                top_sentiment = max(sentiment_result, key=lambda x: x['score']) # Get the sentiment with the higher score
                tweets_analysis.append(top_sentiment['label'])
        
            except Exception as e:
                print(e)

        return np.median(tweets_analysis)

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

            median_sentiment = self.extract_median_tweet_sentiment(tweets["text"])

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