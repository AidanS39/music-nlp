import random

def train_test_split(data: list(), train_split: float):
    # generate the random train test indices over the data set
    train_indices = random.sample(range(len(data)), (int)(len(data) * train_split))
    test_indices = [i for i in range(len(data)) if i not in train_indices]
    
    # generate the train and test sets
    train = [data[i] for i in train_indices]
    test = [data[i] for i in test_indices]

    return train, test
