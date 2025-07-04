import math

def count_n_grams(documents: list(), n_gram: int):
    
    # dictionary of n-gram counts for every classifier
    counts = dict()
    
    # set of all words in dataset
    vocab = set()
    
    # dictionary of document counts for every classifier
    num_docs = dict()
   
    # set of classifiers
    classifiers = set()
     
    # iterate through all documents
    for doc in documents:
        words = doc[1].split()
        word_count = len(words)
        classifier = doc[0]  
        classifiers.add(classifier)
 
        # initialize classifier n-grams, vocab, word count, and number of docs if classifier not seen yet
        if classifier not in counts:
            counts[classifier] = dict()
        
        counts[classifier][0] = counts[classifier].get(0, 0) + word_count
        num_docs[classifier] = num_docs.get(classifier, 0) + 1
        vocab.update(words)
        
        # iterate through every n-gram from 1 to n_gram
        for n in range(1, n_gram + 1):
            if n > word_count:
                break
            if n not in counts[classifier]:
                counts[classifier][n] = dict()

            # iterate through every n-gram sequence in document for current n
            for i in range(word_count - n + 1):
                gram = tuple(words[i:i+n])
                counts[classifier][n][gram] = counts[classifier][n].get(gram, 0) + 1
    
    return counts, vocab, num_docs, classifiers 

def train_n_grams(counts: dict(), vocab: dict(), classifiers: set(), num_docs: dict(), n: int):
    
    # dictionary of log n-gram likelihoods for each classifier
    log_likelihoods = dict()
    
    # dictionary of log prior likelihoods for each classifier 
    log_priors = dict()
    
    # total number of documents in training set
    total_docs = sum([num_docs[classifier] for classifier in num_docs])     

    # size of vocabulary
    vocab_size = len(vocab)
 
    for classifier in classifiers:
        log_priors[classifier] = math.log2(num_docs[classifier]) - math.log2(total_docs)
        log_likelihoods[classifier] = dict()
        
        # smoothed likelihood for unseen words
        log_likelihoods[classifier][0] = math.log2(1) - math.log2(counts[classifier][0] + vocab_size)
        
        for i in range(1, n + 1):
            log_likelihoods[classifier][i] = dict()
            for gram in counts[classifier][i]:
                if i > 1:
                    log_likelihoods[classifier][i][gram] = math.log2(counts[classifier][i][gram]) - math.log2(counts[classifier][i-1][gram[:i-1]])
                else:
                    log_likelihoods[classifier][i][gram] = math.log2(counts[classifier][i][gram]) - math.log2(counts[classifier][i-1])
                    
    return log_likelihoods, log_priors, classifiers, vocab

def test_n_grams(model: tuple(), document: list(), n: int):
    log_likelihoods = model[0]
    log_priors = model[1]
    classifiers = model[2]
    vocab = model[3]

    vocab_size = len(vocab)
    
    probabilities = list()
    
    
    # iterate through all classifiers, calculate each probability given document
    for classifier in classifiers:
        probability = log_priors[classifier]
        for i in range(len(document) - n + 1): 
            gram = tuple(document[i:i+n])
            temp_gram = gram
            discount = 0
            k = n
            # (stupid) backoff to lower n-gram if current n-gram does not exist
            while k >= 1 and gram not in log_likelihoods[classifier][k]:
                k -= 1
                discount += math.log2(0.4)
                temp_gram = temp_gram[1:]
            # add likelihood of each n-gram, or 0 if backed off past unigram (word does not exist in vocab)
            if k >= 1:
                if temp_gram in log_likelihoods[classifier][k]:
                    probability += discount + log_likelihoods[classifier][k][temp_gram]
            else:
                probability += discount + log_likelihoods[classifier][0]
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
     
def print_test_n_grams(n_gram_model: tuple(), document: list(), n: int):
    probs = test_n_grams(n_gram_model, document[1].split(), n)
    
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

def top_k_evaluation(model: tuple(), test_set: list(), k: int, n: int):
    total_documents = len(test_set)
    correct_documents = 0
    for document in test_set:
        ranked_classifiers = [classifier[0] for i, classifier in enumerate(test_n_grams(model, document[1].split(), n), 1) if i <= k]
        if document[0] in ranked_classifiers:
            correct_documents += 1
    print(str(correct_documents) + "/" + str(total_documents) + " in top " + str(k))

def top_k_classifier_evaluation(model: tuple(), test_set: list(), k: int, classifier: str, n: int):
    classifier_set = [document for document in test_set if document[0] == classifier]
    total_documents = len(classifier_set)
    correct_documents = 0
    for document in classifier_set:
        ranked_classifiers = [classifier[0] for i, classifier in enumerate(test_n_grams(model, document[1].split(), n), 1) if i <= k]
        if document[0] in ranked_classifiers:
            correct_documents += 1
    print("For " + classifier + ":")
    print(str(correct_documents) + "/" + str(total_documents) + " in top " + str(k))

# NOTE: does not work without huge amounts of memory, since every possible n_gram probability from the vocab is being calculated.
# generalized training n-grams with backoff smoothing
def train_ALL_n_grams(documents: list(), n_gram: int):
    
    counts, vocab, total_lengths = count_n_grams(documents, n_gram) 
    n_grams = dict()
    
    # calculate smoothed n-gram probabilities
    for classifier in counts:
        n_grams[classifier] = dict()
        vocab_list = list(vocab[classifier])
        vocab_size = len(vocab[classifier])
        pos = dict()
        for n in range(n_gram):
            pos[n] = 0
        
        # dynamically iterate through all n-grams
        while pos[0] < vocab_size:
            gram = tuple(vocab_list[pos[n]] for n in range(n_gram))
            temp_gram = gram
            i = n_gram
            discount = 1
            
            # backoff to n-1 if gram has count of 0
            while i > 0 and temp_gram not in counts[classifier][i]:
                temp_gram = temp_gram[1:]
                i -= 1

                # stupid backoff discount value
                discount *= 0.4
            if i < 1:
                print("Error: backed off past unigram when modeling")
                print(gram)
                print(classifier)
                return -1
            elif i == 1:
                n_grams[classifier][gram] = discount * (counts[classifier][i][temp_gram] / total_lengths[classifier])
            else:
                n_grams[classifier][gram] = discount * (counts[classifier][i][temp_gram] / counts[classifier][i][temp_gram[:i]])
            
            # increment iterator 
            i = n_gram - 1
            pos[i] += 1
            while i >= 1 and pos[i] >= vocab_size:
                pos[i] = 0
                i -= 1
                pos[i] += 1
    
        # attempt to save memory
        counts[classifier].clear()
    
    return n_grams

