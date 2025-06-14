import preprocessing
import math
import random

def train_test_split(data: list(), train_split: float):
    # generate the random train test indices over the data set
    train_indices = random.sample(range(len(data)), (int)(len(data) * train_split))
    test_indices = [i for i in range(len(data)) if i not in train_indices]
    
    # generate the train and test sets
    train = [data[i] for i in train_indices]
    test = [data[i] for i in test_indices]

    return train, test

# NOTE: documents should be a list of 2-tuples (classifier, content)
def get_naive_elements(documents: list()):
    
    # list of classifiers
    classifiers = list()

    # dictionary of the number of documents for each classifier
    num_docs = dict()
    
    # dictionary of the total number of words of every document for each classifier
    num_words = dict()

    # dictionary of bag of words for each classifier
    bows = dict()

    # set of all words in data
    vocab = set()
    
    # iterate through every document
    for doc in documents:
        classifier = doc[0]
        content = doc[1]
        if classifier not in classifiers:
            classifiers.append(classifier)
            num_docs[classifier] = 1
            num_words[classifier] = 0
            bows[classifier] = dict()
        else:
            num_docs[classifier] += 1

        # iterate through every word in current document
        for word in content.split():
            num_words[classifier] += 1
            vocab.add(word)
            if word not in bows[classifier].keys():
                bows[classifier][word] = 1
            else:
                bows[classifier][word] += 1
    
    return classifiers, num_docs, num_words, bows, vocab

# returns a naive model (classifiers, log prior, log likelihood, vocab) for each classifier
def train_naive_bayes(documents: list()):
    classifiers, num_docs, num_words, bows, vocab = get_naive_elements(documents)
    
    log_priors = dict()
    log_likelihoods = dict()

    total_docs = sum(num_docs.values())
    
    # iterate through all classifiers
    for classifier in classifiers:
        log_priors[classifier] = math.log2(num_docs[classifier] / total_docs)
        log_likelihoods[classifier] = dict()
        
        # iterate through all words in classifier's bag of words
        for word in vocab:
            if word in bows[classifier].keys():
                log_likelihoods[classifier][word] = math.log2((bows[classifier][word] + 1) / (num_words[classifier] + len(vocab)))
            else:
                log_likelihoods[classifier][word] = math.log2((1) / (num_words[classifier] + len(vocab)))
    
    return (classifiers, log_priors, log_likelihoods, vocab)

# returns a sorted dictionary of classes and their relative likeliness given the document
def test_naive_bayes(naive_model: tuple(), document: list()):
    classifiers = naive_model[0]
    log_priors = naive_model[1]
    log_likelihoods = naive_model[2]
    vocab = naive_model[3]
    
    probabilities = list()
    
    # calculate first classifier's probability add as a baseline to probabilities for sorting
    probability = log_priors[classifiers[0]]
    
    for word in document:
        if word in vocab:
            probability += log_likelihoods[classifiers[0]][word]
    
    probabilities.append((classifiers[0], probability))

    # iterate through all classifiers, calculate each probability given document
    for classifier in classifiers[1:]:
        probability = log_priors[classifier]
        for word in document:
            if word in vocab:
                probability += log_likelihoods[classifier][word]
        
        # find ordered index and insert current probability to probabilities
        i = 0
        for prob in probabilities:
            if probability > prob[1]:
                probabilities.insert(i, (classifier, probability))
                break
            else:
                i += 1
        if i >= len(probabilities):
            probabilities.append((classifier, probability))
 
    return probabilities

def print_test_naive_bayes(naive_model: tuple(), document: list()):
    probs = test_naive_bayes(naive_model, document[1].split())
    
    print("Song Lyrics: ")
    print(document[1])

    print("Most likely artist")
    print("----------------------")
    print("Actual Artist: " + document[0])
    print("Song Title: " + document[2])
    print("----------------------")
    print(f"{'Rankings':<10}{'Artist':<20}{'Log Probability':<20}")
    for i, prob in enumerate(probs):
       print(f"{str(i + 1):<10}{prob[0]:<20}{prob[1]:<20}") 

def top_k_evaluation(naive_model: tuple(), test_set: list(), k: int):
    total_documents = len(test_set)
    correct_documents = 0
    for document in test_set:
        ranked_classifiers = [classifier[0] for i, classifier in enumerate(test_naive_bayes(naive_model, document[1].split()), 1) if i <= k]
        if document[0] in ranked_classifiers:
            correct_documents += 1
    print(str(correct_documents) + "/" + str(total_documents) + " in top " + str(k))

def main():
    print("Preprocessing song data...")
    songs = preprocessing.song_lyrics_dataset()

    print("Splitting data into training and test sets...")
    train, test = train_test_split(songs, 0.9)

    print("Training model...")
    model = train_naive_bayes(train)
    
    print("Welcome to the Naive Bayes Music Program")

    while True:
        print("----------------------")
        print("Options:")        
        print("(1)        Most likely artist for specified song")        
        print("(2)        Top k Evaluation")
        print("([ENTER])  Exit")
        print("----------------------")
        
        option = input("Please select an option: ")
        if option == "1":
            try:
                song_index = int(input(f"Please select a song index from 1 to {len(test)}: "))
            except ValueError:
                print("Invalid song index.")
            if song_index > 0 and song_index <= len(test):
                print_test_naive_bayes(model, test[song_index - 1])
            else:
                print("Invalid song index. Song index must be a number from 1 to {len(test)}: ")
        elif option == "2": 
            try:
                k = int(input(f"Please select a k from 1 to {len(model[0])}: "))
            except ValueError:
                print("Invalid k.")
            if k > 0 and k <= len(model[0]):
                top_k_evaluation(model, test, k)
            else:
                print("Invalid k. k must be a number from 1 to {len(model[0])}")
        elif option == "":
            return 0
        else:
            print("Not a valid option.")

main()
