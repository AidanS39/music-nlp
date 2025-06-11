import preprocessing
import math

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
            if word not in bows[classifier].keys():
                bows[classifier][word] = 1
            else:
                bows[classifier][word] += 1

    return classifiers, num_docs, num_words, bows

# returns a naive model (classifiers, log prior, log likelihood) for each classifier
def train_naive_bayes(documents: list()):
    classifiers, num_docs, num_words, bows = get_naive_elements(documents)
    
    log_priors = dict()
    log_likelihoods = dict()

    total_docs = sum(num_docs.values())
    
    # iterate through all classifiers
    for classifier in classifiers:
        log_priors[classifier] = math.log2(num_docs[classifier] / total_docs)
        log_likelihoods[classifier] = dict()
        
        # iterate through all words in classifier's bag of words
        for word in bows[classifier]:
            log_likelihoods[classifier][word] = math.log2(bows[classifier][word] / num_words[classifier])
    
    return (classifiers, log_priors, log_likelihoods)

# returns the most likely class given the document
def test_naive_bayes(naive_model: tuple(), document: list()):
    classifiers = naive_model[0]
    log_priors = naive_model[1]
    log_likelihoods = naive_model[2]
    
    # initialize first classifier as the maximum a posterior 
    c_map = classifiers[0]
    maximum = log_priors[classifiers[0]] + sum([log_likelihoods[classifiers[0]][word] for word in document if word in log_likelihoods[classifiers[0]]])
    
    # iterate through all classifiers, find the maximum probability given document
    for classifier in classifiers:
        probability = log_priors[classifier] + sum([log_likelihoods[classifier][word] for word in document if word in log_likelihoods[classifier]])
        if probability > maximum:
            maximum = probability
            c_map = classifier

    return c_map


songs = preprocessing.song_lyrics_dataset()

model = train_naive_bayes(songs)


print(test_naive_bayes(model, ["come", "up", "to", "meet", "you", "tell", "you", "sorry"]))

