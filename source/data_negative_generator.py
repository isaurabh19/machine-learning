from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import json

SPOTIFY_CLIENT_ID = "a99f3cb2bc5245f4a7693a7fb422078c"
SPOTIFY_CLIENT_SECRET = "87bcf9260b734bd79fe50c93e0778830"


def dfs_crawl(seed, url_list):
    if seed == '':
        return
    if len(url_list) == 0:
        url_list.append(seed)
    # Getting the page using requests
    response = requests.get('https://en.wikipedia.org' + seed)
    if response.status_code == 301 or response.status_code == 302:
        print("Redirected URL")

    page_text = response.text

    tags = BeautifulSoup(page_text, "html.parser")

    for tag in tags.find_all('a'):
        if tag.text == 'next page' and tag['href'] not in url_list:
            url_list.append(tag['href'])
            dfs_crawl(tag['href'], url_list)
    return url_list

# Method to crawl a page


def page_crawl(url, song_list):

    # song_list = []

    # Getting the page using requests
    response = requests.get('https://en.wikipedia.org' + url)
    if response.status_code == 301 or response.status_code == 302:
        print("Redirected URL")

    page_text = response.text

    tags = BeautifulSoup(page_text, "html.parser")

    for tag in tags.find_all("div", {"class": "mw-category-group"}):
        for t in tag.find_all("a", href=re.compile("^/wiki/")):
            title = t.get("title")
            title = re.sub(r'\([^)]*\)', '', title)
            song_list.append(title + "\n")
    return song_list


def get_song_ids(song_list):
    song_ids = []
    client_credentials_manager = SpotifyClientCredentials(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    for song in song_list:
        print('Song: ', song, '\n')
        search = sp.search(q=song, type='track', limit=1)
        if len(search['tracks']['items']) != 0:
            song_ids.append(search['tracks']['items'][0]['id'])
    return np.asanyarray(song_ids)


def get_playlist_tracks(playlist_id, ids):
    client_credentials_manager = SpotifyClientCredentials(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    tracks = sp.playlist_tracks(playlist_id=playlist_id)
    if len(tracks['items']) != 0:
        for i in range(len(tracks['items'])):
            ids.append(tracks['items'][i]['track']['id'])
    if tracks['next']:
        while tracks['next']:
            tracks = sp.next(tracks)
            # print(len(tracks['items']))
            if len(tracks['items']) != 0:
                for i in range(len(tracks['items'])):
                    ids.append(tracks['items'][i]['track']['id'])
    print(len(ids), '\n')
    return ids


if __name__ == "__main__":
    playlists = ['35GvQpyFcVqPjulIhRvKeW', '2JpL5Q4XH711AwgxK9YwM0', '4SvPa1JBBT9qxp2vkEnJ2S', '1ilzM4vhx25DPNJpP2XXUM']
    track_ids = []
    for p_id in playlists:
        track_ids = get_playlist_tracks(p_id, track_ids)
    df_track_ids = pd.DataFrame(data=track_ids)
    df_track_ids.to_csv('playlist.csv')
    # urls = dfs_crawl('/wiki/Category:2020_songs', [])
    # song_list = []
    # for url in urls:
    #     song_list = page_crawl(url, song_list)
    # ids = get_song_ids(song_list)
    # df = pd.DataFrame(data=ids)
    # df.to_csv('2020.csv')


# 35GvQpyFcVqPjulIhRvKeW
# 2JpL5Q4XH711AwgxK9YwM0
# 4SvPa1JBBT9qxp2vkEnJ2S
# 1ilzM4vhx25DPNJpP2XXUM
