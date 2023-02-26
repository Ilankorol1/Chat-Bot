import random
import json
import pickle
import numpy as np
from nltk.corpus import stopwords
import re

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize_removeStopWord(query):
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens_list = []
    no_dupl = list(dict.fromkeys(tokens))
    for t in no_dupl:
        if t not in all_stopwords:
            tokens_list.append(t)
    return tokens_list


lematizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = tokenize_removeStopWord(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

wnl = WordNetLemmatizer()
words = [wnl.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(words, open('classes.pkl','wb'))

training = []
output_empy = [0]* len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lematizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if(word in word_patterns) else bag.append(0)
    output_row = list(output_empy)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[: , 0])
train_y = list(training[: , 1])

model = Sequential()
