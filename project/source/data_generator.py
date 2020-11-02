from fycharts.SpotifyCharts import SpotifyCharts
from joblib import Parallel, delayed
import glob
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import pprint

SPOTIFY_CLIENT_ID = "whatever_your_creds"
SPOTIFY_CLIENT_SECRET = "secret"


def get_data(date_range):
    print("Dates are {} {}".format(date_range[0], date_range[1]))
    charts_api = SpotifyCharts()
    charts_api.top200Weekly("top200weekly" + date_range[0] + ".csv", start=date_range[0], end=date_range[1],
                            region=["us"])


def parallel_getcsv():
    date_ranges = [("2017-01-06", "2017-06-01"), ("2017-06-02", "2017-11-02"), ("2017-11-03", "2018-04-05"),
                   ("2018-04-06", "2018-09-06"), ("2018-09-07", "2019-02-07"), ("2019-02-08", "2019-07-04"),
                   ("2019-07-05", "2019-12-05"), ("2019-12-06", "2020-03-26")]

    Parallel(n_jobs=8, verbose=10)(delayed(get_data)(dates) for dates in date_ranges)


def collect_attributes_for_song(ids_list):
    start = 0
    end = 50
    features_list = []
    total_songs = len(ids_list)
    counter = 0
    while start < total_songs:
        tracks = ids_list[start:end]
        features = sp.audio_features(tracks)
        features_list.extend(features)
        counter += end - start
        # print("Done {}/{}".format(counter, total_songs))
        start += 50
        end = end + 50 if end + 50 <= total_songs else total_songs

    features_df = pd.DataFrame(features_list)
    features_df.to_csv("..//data/features.csv")
    return features_df


def get_spotify_client():
    client_credentials_manager = SpotifyClientCredentials(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

def take_top_n(csv):
    df = pd.read_csv(csv)
    return df[df['Position'] <= 30.]

def merge_csv():
    all_files = glob.glob("..//data/top200weekly*.csv")
    df_merged = pd.concat([take_top_n(f) for f in all_files])
    df_merged.reset_index(drop=True, inplace=True)
    df_merged.to_csv("..//data/top_n_positive_dataset.csv")


def remove_hit_songs(csv, positive_song_df):
    negative_df = pd.read_csv(csv, index_col=0)
    negative_df = negative_df.drop_duplicates(subset='spotify_id')
    negative_df = negative_df[~negative_df['spotify_id'].isin(positive_song_df['spotify_id'])]
    return negative_df['spotify_id']


def get_negative_features(positive_song_df):
    # List of Series
    negative_ids = Parallel(n_jobs=8, verbose=1)(
        delayed(remove_hit_songs)(csv, positive_song_df) for csv in glob.glob("20*.csv"))
    negative_featues = [collect_attributes_for_song(series.tolist())for series in negative_ids]#list(map(collect_attributes_for_song, negative_ids))
    negative_dataset_df = pd.concat(negative_featues, ignore_index=True, axis=0)
    negative_dataset_df = process_features_df(negative_dataset_df)
    return negative_dataset_df


def process_features_df(features_df):
    columns_to_remove = ["type", "uri", "track_href", "analysis_url"]
    features_df.drop(columns=columns_to_remove, inplace=True)
    dataset_df = features_df.set_index("id")
    dataset_df.dropna(inplace=True)
    return dataset_df

def get_positive_features(song_ids):
    # features_list = Parallel(n_jobs=8, verbose=5)(
    #     delayed(sp.audio_features)(spotify_id) for index,spotify_id in song_ids.items())
    # features_df = collect_attributes_for_song(song_ids.to_numpy())
    features_df = pd.read_csv("..//data/features.csv", index_col=0)
    positive_dataset_df = process_features_df(features_df)
    positive_dataset_df.to_csv('../data/top_n_final_pos_features.csv')
    return positive_dataset_df

class DataGenerator:
    dataset_df = None

    def get_dataset(self):
        pos_df = pd.read_csv("../data/top_n_final_pos_features.csv")
        neg_df = pd.read_csv("../data/final_neg_features.csv")
        length_neg = len(neg_df.columns)
        length_pos = len(pos_df.columns)
        assert length_pos == length_neg

        pos_df.insert(length_pos, column='label', value=1)
        neg_df.insert(length_neg, column='label', value=0)

        dataset_df = pd.concat([pos_df, neg_df], ignore_index=True, axis=0)
        # self.dataset_df = self.dataset_df.apply(np.random.permutation, axis=0)

        return dataset_df

    def get_separate_datasets(self):
        pos_df = pd.read_csv("../data/final_pos_features.csv")
        neg_df = pd.read_csv("../data/final_neg_features.csv")
        return pos_df, neg_df

    def convert_to_numpy_dataset(self, df):
        df = df.to_numpy()
        return df[:, 1:]


# 1. Setup spotify client
# 2. Parrallelize song feature extraction

# parallel_getcsv()
# merge_csv()
# sp = get_spotify_client()
# positive_df = pd.read_csv("..//data/top_n_positive_dataset.csv", delimiter=",", index_col=0)
# unique_songs = positive_df.drop_duplicates(subset='spotify_id')
# song_ids = unique_songs['spotify_id']
# song_ids.dropna(inplace=True)
# # positive_df = get_positive_features(song_ids)
# negative_dataset_df = get_negative_features(unique_songs)
