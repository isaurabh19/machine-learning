from fycharts.SpotifyCharts import SpotifyCharts
from joblib import Parallel, delayed
import glob
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import pprint

SPOTIFY_CLIENT_ID = "a99f3cb2bc5245f4a7693a7fb422078c"
SPOTIFY_CLIENT_SECRET = "87bcf9260b734bd79fe50c93e0778830"


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
        start +=50
        end = end + 50 if end + 50 <= total_songs else total_songs

    features_df = pd.DataFrame(features_list)
    features_df.to_csv("..//data/features.csv")
    return features_df

def get_spotify_client():
    client_credentials_manager = SpotifyClientCredentials(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp


def merge_csv():
    all_files = glob.glob("..//data/*.csv")
    df_merged = pd.concat([pd.read_csv(f) for f in all_files])
    df_merged.reset_index(drop=True, inplace=True)
    df_merged.to_csv("..//data/positive_dataset.csv")


# 1. Setup spotify client
# 2. Parrallelize song feature extraction

# parallel_getcsv()
# merge_csv()
positive_df = pd.read_csv("..//data/positive_dataset.csv",delimiter=",", index_col=0)
unique_songs = positive_df.drop_duplicates(subset='spotify_id')
song_ids = unique_songs['spotify_id']
song_ids.dropna(inplace=True)
sp = get_spotify_client()
# features_list = Parallel(n_jobs=8, verbose=5)(
#     delayed(sp.audio_features)(spotify_id) for index,spotify_id in song_ids.items())
features_df = pd.read_csv("..//data/features.csv", index_col=0)#collect_attributes_for_song(song_ids.tolist())
columns_to_remove = ["type","uri","track_href","analysis_url"]
features_df.drop(columns=columns_to_remove, inplace=True)
positive_dataset_df = features_df.set_index("id")
print(positive_dataset_df)