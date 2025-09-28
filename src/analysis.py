import preprocessing
import naive_bayes
import n_grams
import util
import math
import csv
import os

def gather_naive_test_results(naive_model, test, num_artists):
    
    test_rankings = []

    # for each test doc, determine actual artist's rank
    for document in test:
        target_artist = document[0]
        ranking = naive_bayes.test_naive_bayes(naive_model, document[1].split())
        
        k = 1
        for rank in ranking:
            artist = rank[0]
            if target_artist == artist:
                test_rankings.append(k)
                break
            else:
                k += 1
        if k > num_artists:
            print(f"error: artist not found in ranking: {target_artist}")
            return -1
    return test_rankings
    
def gather_n_grams_test_results(n_gram_model, test, num_artists, n):
    
    test_rankings = []

    # for each test doc, determine actual artist's rank
    for document in test:
        target_artist = document[0]
        ranking = n_grams.test_n_grams(n_gram_model, document[1].split(), n)
        
        k = 1
        for rank in ranking:
            artist = rank[0]
            if target_artist == artist:
                test_rankings.append(k)
                break
            else:
                k += 1
        if k > num_artists:
            print(f"error: artist not found in ranking: {target_artist}")
            return -1
    return test_rankings

def naive_analysis(train, test):
    naive_model = naive_bayes.train_naive_bayes(train)
    
    artists = naive_model[0]    

    test_rankings = gather_naive_test_results(naive_model, test, len(artists))
    
    # count top-k data
    top_k = {k: 0 for k in range(1, len(artists) + 1)}
    
    for ranking in test_rankings:
        for k in range(len(artists), 0, -1):
            if ranking <= k:
                top_k[k] += 1
            elif ranking > k: # short circuit if k is past result
                break
    
    file_name = "top_k_naive_results.csv" 
    if os.path.exists(file_name) == False:
        with open(file_name, "w") as file:
            writer = csv.writer(file)
            headers = [f"{k}" for k in range(1, len(artists) + 1)]
            writer.writerow(headers)
    
    with open(file_name, "a") as file:
        writer = csv.writer(file)
        writer.writerow(list(top_k.values()))

def n_grams_analysis(train, test, n):
    counts, vocab, num_docs, artists = n_grams.count_n_grams(train, n)
    n_gram_model = n_grams.train_n_grams(counts, vocab, artists, num_docs, n)
    
    test_rankings = gather_n_grams_test_results(n_gram_model, test, len(artists), n)
    
    # count top-k data
    top_k = {k: 0 for k in range(1, len(artists) + 1)}
    
    for ranking in test_rankings:
        for k in range(len(artists), 0, -1):
            if ranking <= k:
                top_k[k] += 1
            elif ranking > k: # short circuit if k is past result
                break
    
    file_name = f"top_k_{n}_grams_results.csv" 
    if os.path.exists(file_name) == False:
        with open(file_name, "w") as file:
            writer = csv.writer(file)
            headers = [f"{k}" for k in range(1, len(artists) + 1)]
            writer.writerow(headers)
    
    with open(file_name, "a") as file:
        writer = csv.writer(file)
        writer.writerow(list(top_k.values()))


def main():
    for i in range(100):
        songs = preprocessing.song_lyrics_dataset()
        train, test = util.train_test_split(songs, 0.9)
    
        n_grams_analysis(train, test, 1)


main()
