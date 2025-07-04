import os
import csv
import re

# returns a list of (artist, lyrics, title) from every song in the Songs Lyrics Dataset
def song_lyrics_dataset():
    songs = list()
    
    dir_path = "../data/song-lyrics-dataset/csv/" 
    for file_name in os.listdir(dir_path):
        file_path = dir_path + file_name
        with open(file_path, "r") as songs_file:
            reader = csv.DictReader(songs_file)
            for song in reader:
                song["Lyric"] = re.sub(r'[^\w\s]', '', song["Lyric"])
                song["Lyric"] = re.sub(r'\s+', ' ', song["Lyric"])
                songs.append((song["Artist"], song["Lyric"], song["Title"]))
    return songs

# NOTE: do not use. function is flawed.
def song_lyrics_dataset_OLD():
    songs = list()
    
    # iterate through every file
    dir_path = "../data/song-lyrics-dataset/csv/" 
    for file_name in os.listdir(dir_path):
        file_path = dir_path + file_name
        with open(file_path, "r") as songs_file:
            
            # read all songs from file 
            cur_songs = songs_file.readlines()
            
            # analyze table header, find index of artist and lyrics
            header = cur_songs[0].rstrip()
            artist_index = -1
            title_index = -1
            lyric_index = -1
            for i, title_col in enumerate(header.split(',')):
                if title_col == "Artist":
                    if artist_index == -1:
                        artist_index = i
                    else:
                        print("Error: multiple Artist columns in file.")
                        return -1
                elif title_col == "Title":
                    if title_index == -1:
                        title_index = i
                    else:
                        print("Error: multiple Title columns in file.")
                        return -1
                elif title_col == "Lyric":
                    if lyric_index == -1:
                        lyric_index = i
                    else:
                        print("Error: multiple Lyric columns in file.")
                        return -1
            
            if artist_index == -1:
                print("Error: Artist column not found.")
                return -1
            if lyric_index == -1:
                print("Error: Lyric column not found.")
                return -1
            if title_index == -1:
                print("Error: Title column not found.")
                return -1


            # add artist and lyrics to list of songs
            for song in cur_songs[1:]:
                song = song.split(',')
                songs.append((song[artist_index], song[lyric_index].rstrip(), song[title_index]))
    return songs

# TODO
def genius_song_lyrics():
    return 0
