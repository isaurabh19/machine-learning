from bs4 import BeautifulSoup
import requests
import re
import numpy as np

# Method to crawl a page


def page_crawl(url):

    song_list = []

    # Getting the page using requests
    response = requests.get(url)
    if response.status_code == 301 or response.status_code == 302:
        print("Redirected URL")

    page_text = response.text

    tags = BeautifulSoup(page_text, "html.parser")

    for tag in tags.find_all("div", {"class": "mw-category-group"}):
        for t in tag.find_all("a", href=re.compile("^/wiki/")):
            title = t.get("title")
            title = re.sub(r'\([^)]*\)', '', title)
            song_list.append(title + "\n")
    return np.asanyarray(song_list)


if __name__ == "__main__":
    songs = page_crawl("https://en.wikipedia.org/wiki/Category:2017_songs")
    print(songs)
    # Code to write to file
    with open(r'test.txt', 'a') as f:
        f.write(" ".join(map(str, songs)))
