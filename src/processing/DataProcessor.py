import locale
import os
from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import src.utils.utilities as utils

BOT_LABEL = 0
HUMAN_LABEL = 1

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
    # "median_day_of_tweeting", # -1 if no tweets
    # "median_time_of_tweeting", # -1 if no tweets
]


class DataProcessor:
    def __init__(self, args: Namespace):
        self.user_file = "users.csv"
        self.tweet_file = "tweets.csv"
        self.args = args
        self.run()

    def run(self) -> None:
        # Set features
        df_human: pd.DataFrame = self._process_dataset(
            self.args.humans_folder, HUMAN_LABEL
        )
        df_bot: pd.DataFrame = self._process_dataset(self.args.bots_folder, BOT_LABEL)

        df_human = self._remove_outliers_by_numerical_cols(df_human)
        df_bot = self._remove_outliers_by_numerical_cols(df_bot)

        # Assert identical lengths and merge
        assert df_human.columns.equals(df_bot.columns)

        self.df_merged: pd.DataFrame = pd.concat([df_human, df_bot])
        self.df_merged = self._process_merged(self.df_merged)

        # Remove non-numeric columns and visualize
        self.visualize(self.df_merged, "feature_correlation")

        # Remove outliers and apply min-max scaler
        # self.data_merged = self._remove_outliers_by_numerical_cols(numeric_df)
        self.df_merged = self.apply_min_max_scaler(self.df_merged)
        self.make_pair_plot()

        train, test = self.get_train_test()

        # save data
        save_dir = os.path.join(os.getcwd(), "data", "processed")
        os.makedirs(save_dir, exist_ok=True)
        train.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(save_dir, "test.csv"), index=False)
        self.check_duplicates(self.df_merged)
        self.df_merged.to_csv(
            os.path.join(save_dir, "data_parsed.csv"), index=False
        )  # unsplit for debugging

    def make_pair_plot(self):
        """Make a pair plot"""
        # use only numeric columns
        output_dir = os.path.join(os.getcwd(), "output", "images", __class__.__name__)

        numeric_columns.append("account_type")
        numeric_df = self.df_merged[numeric_columns]
        plt.figure(figsize=(20, 20))
        sns.pairplot(numeric_df, hue="account_type")
        plt.savefig(os.path.join(output_dir, "pairplot.png"))
        plt.clf()
        plt.close()

    def check_duplicates(self, df: pd.DataFrame):
        """Check if there are any duplicates"""
        print(df.duplicated().sum())

    def get_train_test(self):
        """Balance the dataset by returning a train and test set
        Train test size should be 80% of the minority class and majority
        class should be downsampled. We're balancing according to "account_type" attribute
        """
        # Separate minority and majority classes
        human_class: pd.DataFrame = self.df_merged[
            self.df_merged["account_type"] == HUMAN_LABEL
        ]
        bot_class: pd.DataFrame = self.df_merged[
            self.df_merged["account_type"] == BOT_LABEL
        ]

        minority_class: pd.DataFrame = (
            human_class if len(human_class) < len(bot_class) else bot_class
        )
        majority_class: pd.DataFrame = (
            human_class if len(human_class) > len(bot_class) else bot_class
        )

        # Set the train size to be 80% of the minority class
        train_size = int(0.8 * len(minority_class))

        # shuffle-up both classes
        minority_class = minority_class.sample(frac=1, random_state=42).reset_index(
            drop=True
        )
        majority_class = majority_class.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        # Split the minority class by train_size and leftover
        minority_train: pd.DataFrame = minority_class[:train_size]
        minority_test: pd.DataFrame = minority_class[train_size:]

        # Split the majority class by train_size and leftover
        majority_train: pd.DataFrame = majority_class[:train_size]
        majority_test: pd.DataFrame = majority_class[train_size:]

        # Merge both classes
        train_set: pd.DataFrame = pd.concat([minority_train, majority_train])
        test_set: pd.DataFrame = pd.concat([minority_test, majority_test])

        # test if train_set has equal distribution of human/bot
        print("Train set distribution:")
        print(train_set["account_type"].value_counts())

        # test if test_set has equal distribution of human/bot
        print("Test set distribution:")
        print(test_set["account_type"].value_counts())

        return train_set, test_set

    def visualize(self, df: pd.DataFrame, filename) -> None:
        # visualize the correlation
        corr: pd.DataFrame = df.corr()
        f, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            round(corr, 2),
            annot=True,
            ax=ax,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.05,
        )
        f.suptitle("Twitter Bot Account Feature Correlation Heatmap", fontsize=14)

        # Data folder
        output_dir = os.path.join(os.getcwd(), "output", "images", __class__.__name__)
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.clf()
        plt.close()

    def _remove_outliers_by_numerical_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers by columns which are floats"""

        # Remove outliers using z-score
        for column in numeric_columns:
            z_scores = stats.zscore(df[column])
            outliers = (z_scores > 3) | (z_scores < -3)
            df = df[~outliers]

        return df

    def apply_min_max_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min max scaler to all numeric columns"""
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        return df

    def _process_merged(self, df: pd.DataFrame) -> pd.DataFrame:
        # # create lang categories
        # data["lang_cat"] = data["lang"].astype("category").cat.codes
        # # one-hot
        # encoded = pd.get_dummies(data["lang_cat"], prefix="lang")
        # data = pd.concat([data, encoded], axis=1)

        # Remove ID
        df = df.drop(columns=["id"])
        df = df.drop(columns=["lang"])
        df = df.drop(columns=["name"])
        df = df.drop(columns=["screen_name"])
        df = df.drop(columns=["time_zone"])
        df = df.drop(columns=["updated"])
        df = df.drop(columns=["dataset"])

        # save
        utils.save_tmp_data(df, "process_merged.csv")
        return df

    def _process_dataset(self, datapath, account_type: int) -> pd.DataFrame:
        df: pd.DataFrame = utils.get_data(os.path.join(datapath, self.user_file))

        # add human/bot label
        df["account_type"] = account_type

        # convert date to datetime64
        df = self._convert_date_to_datetime64(df)

        # one-hot encode "url"
        df = self._one_hot_column(df, "url")

        # is "location" present?
        df = self._one_hot_column(df, "location")

        # is "default_profile" present? Add as 0 or 1
        df = self._one_hot_column(df, "default_profile")

        # is "geo_enabled" present?
        df = self._one_hot_column(df, "geo_enabled")

        # drop "profile_image_url"
        df = df.drop(columns=["profile_image_url"])

        # is "profile_banner_url" not "NULL"?
        df = self._one_hot_column(df, "profile_banner_url")

        # is "profile_use_background_image" present?
        df = self._one_hot_column(df, "profile_use_background_image")

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
            "default_profile_image",
        ]

        # drop all columns at once
        df = df.drop(columns=columns_to_drop)

        # add "." to descriptions which are empty
        df["description"] = df["description"].fillna(".")

        # "description" number of characters
        df["description_num_char"] = df["description"].str.len()

        # "description" contains http:// or https:// ?
        df["description"] = df["description"].str.contains("http://|https://")
        df_encoded = pd.get_dummies(
            df["description"], prefix="description_contains_link"
        )
        df = pd.concat([df, df_encoded], axis=1)

        df = self.parse_tweets(df, datapath)

        # save tmp data
        data_name = "human" if account_type is HUMAN_LABEL else "bot"
        utils.save_tmp_data(df, f"process_dataset_{data_name}.csv")

        return df

    def _one_hot_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """One-hot encode a column"""
        # one-hot encode "url"
        df_encoded = pd.get_dummies(
            df[column].notnull().astype(int), prefix=f"{column}_present"
        )
        df = pd.concat([df, df_encoded], axis=1)
        # remove the original column
        df = df.drop(columns=[column])

        return df

    def parse_tweets(self, users_df, datapath):
        """For both users.csv files, map the user IDs to the tweets.csv
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
            """Extract all tweets by user_id
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
            tweets = self._convert_date_to_datetime64(tweets, save=False)

            # add time of day as 1 to 24 as a feature
            tweets = self.add_time_of_day(tweets, save=False)

            # add day of week as 1 to 7 as a feature
            tweets = self.add_day_of_week(tweets, save=False)

            # extract median day of tweeting
            median_day_of_week = tweets["created_day_of_week"].median()
            median_day_of_week = (
                median_day_of_week if pd.notna(median_day_of_week) else -1
            )

            # extract median time of tweeting
            median_time_of_day = tweets["created_time_of_day"].median()
            median_time_of_day = (
                median_time_of_day if pd.notna(median_time_of_day) else -1
            )

            return (
                avg_retweets,
                avg_favorites,
                avg_length,
            )  # , median_day_of_week, median_time_of_day

        users_df[["avg_retweets", "avg_favorites", "avg_length"]] = users_df[
            "id"
        ].apply(lambda x: pd.Series(parse_by_user_id(self, x, tweets_df)))

        # save df
        utils.save_tmp_data(users_df, "parse_tweets.csv")

        return users_df

    def discard_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discard columns that we will not use
        TODO: maybe discard "screen_name" and "id"?
        """
        columns = ["_idx", "id", "profile_background_image_url", "profile_image_url"]
        df = df.drop(columns=columns)
        utils.save_tmp_data(df, "discard_columns.csv")
        return df

    def replace_missing_values(self, df: pd.DataFrame, column, value) -> pd.DataFrame:
        """Replace missing values with the mean of the column"""
        df[column] = df[column].fillna(value)
        utils.save_tmp_data(df, "replace_missing_values.csv")
        return df

    def _convert_date_to_datetime64(self, df: pd.DataFrame, save=True) -> pd.DataFrame:
        """Convert date to datetime64. Necessary for Pycaret."""
        locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
        df = df.assign(
            created_at=lambda x: pd.to_datetime(
                x["created_at"], format="%a %b %d %H:%M:%S +0000 %Y"
            )
        )

        if save:
            utils.save_tmp_data(df, "convert_date_to_datetime64.csv")
        return df

    def add_time_of_day(self, df: pd.DataFrame, save=True) -> pd.DataFrame:
        """Add time of day as 1 to 24 as a feature"""
        df = df.assign(created_time_of_day=lambda x: x["created_at"].dt.hour)

        if save:
            utils.save_tmp_data(df, "add_time_of_day.csv")
        return df

    def add_day_of_week(self, df: pd.DataFrame, save=True) -> pd.DataFrame:
        """Add day of week as 1 to 7 as a feature"""
        df = df.assign(created_day_of_week=lambda x: x["created_at"].dt.dayofweek + 1)

        if save:
            utils.save_tmp_data(df, "add_day_of_week.csv")
        return df

    def convert_account_type_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert account type to int"""
        df = df.assign(
            account_type_int=lambda x: x["account_type"].map({"human": 0, "bot": 1})
        )
        utils.save_tmp_data(df, "convert_account_type_to_int.csv")
        return df
