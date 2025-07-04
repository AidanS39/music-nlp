import preprocessing
import naive_bayes
import n_grams
import math
import random
import time

def train_test_split(data: list(), train_split: float):
    # generate the random train test indices over the data set
    train_indices = random.sample(range(len(data)), (int)(len(data) * train_split))
    test_indices = [i for i in range(len(data)) if i not in train_indices]
    
    # generate the train and test sets
    train = [data[i] for i in train_indices]
    test = [data[i] for i in test_indices]

    return train, test

def user_input_artist(artists: list()):
    print("Artists: ") 
    for i, artist in enumerate(artists):
        print(f"({i + 1})\t{artist}")
    try:
        artist_index = int(input("\nPlease choose an artist by their index: "))
    except ValueError:
        print("Invalid artist index.")
        return -1 
    if artist_index < 1 or artist_index > len(artists):
        print("Invalid artist index.")
        return -1 
    else:
        return artists[artist_index - 1]

def user_input_song_by_artist(artist: str, songs: list()):
    artist_song_list = [(song[2], songs_index) for songs_index, song in enumerate(songs) if song[0] == artist]
    
    print(f"Songs by {artist}:")
    for i, song in enumerate(artist_song_list):
        print(f"{i + 1}\t\t{song[0]}") 
    
    try:
        song_index = int(input(f"Please select a song by its index: "))
    except ValueError:
        print("Invalid song index.")
        return -1
    if song_index < 0 and song_index > len(artist_song_list):
        print("Invalid song index. Song index must be a number from 1 to {len(artist_song_list)}: ")
        return -1
    else:
        return songs[artist_song_list[song_index - 1][1]]

def user_input_k(max_k: int):
    try:
        k = int(input(f"Please select a k from 1 to {max_k}: "))
    except ValueError:
        print("Invalid k.")
        return -1
    if k > 0 and k <= max_k:
        return k
    else:
        print(f"Invalid k. k must be a number from 1 to {max_k}")
        return -1

def main():
    print("Preprocessing song data...")
    songs = preprocessing.song_lyrics_dataset()

    print("Splitting data into training and test sets...")
    train, test = train_test_split(songs, 0.9)

    print("\nWelcome to the NLP Music Program")
    while True: 
        print("----------------------")
        print("Options:")        
        print("(1)            Naive Bayes")        
        print("(2)            N-Grams")
        print("----------------------")
            
        model_option = input("Please select a model: ")
        if model_option not in ["1", "2"]:
            print("Not a valid option.")
            input("Press [ENTER] to continue.")
        else:
            break
    if model_option == "1":
        print("Training model...")
        model = naive_bayes.train_naive_bayes(train) 
        artists = model[0]
    elif model_option == "2":
        while True:
            n = int(input("Enter an n: "))
            if n < 1:
                print("Invalid option.")
                input("Press [ENTER] to continue.")
            else:
                break
        print("Counting n-grams...")
        counts, vocab, num_docs, classifiers = n_grams.count_n_grams(train, n)
        print("Calculating n-gram probabilities...")
        model = n_grams.train_n_grams(counts, vocab, classifiers, num_docs, n)
        artists = list(classifiers)
    else:
        print("Error: invalid model option.")
        return
     
    while True:
        print("----------------------")
        print("Options:")        
        print("(1)        Most likely artist for specified song")        
        print("(2)        Top k Evaluation")
        print("(3)        Top k Artist Evaluation")
        print("([ENTER])  Exit")
        print("----------------------")
        
        option = input("Please select an option: ")
        if option == "1":
            artist = user_input_artist(artists)
            if artist == -1:
                continue
            song = user_input_song_by_artist(artist, test)
            if song == -1:
                continue
            if model_option == "1":
                naive_bayes.print_test_naive_bayes(model, song)
            elif model_option == "2":
                n_grams.print_test_n_grams(model, song, n)
        elif option == "2":
            k = user_input_k(len(artists)) 
            if k == -1:
                continue 
            if model_option == "1":
                naive_bayes.top_k_evaluation(model, test, k)
            elif model_option == "2":
                n_grams.top_k_evaluation(model, test, k, n)
        elif option == "3":
            artist = user_input_artist(artists) 
            if artist == -1:
                continue
            k = user_input_k(len(artists)) 
            if k == -1:
                continue
            if model_option == "1":
                naive_bayes.top_k_classifier_evaluation(model, test, k, artist)
            elif model_option == "2":
                n_grams.top_k_classifier_evaluation(model, test, k, artist, n)
        elif option == "":
            return 0
        else:
            print("Not a valid option.")
        input("Press [ENTER] to continue.")

main()

