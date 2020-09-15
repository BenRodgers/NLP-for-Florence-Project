import json
import string

import gensim
import nltk
from flask import Flask, request, jsonify
from gensim import corpora
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

'''
    Function for getting the complexity of the vocabulary used from the text (using Oxford CEFR)
'''


def get_vocabulary_complexity(conversation):
    vocab_complexity = {'simple': 0, 'complex': 0}

    conversation = conversation.lower()
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
    conversation = conversation.translate(remove_punct_map)
    conversation = conversation.split()
    with open("./wordlist/Oxford 3000.txt", "r") as f:
        for line in f:
            for i in conversation:
                if i+"\n" == line:
                    vocab_complexity['simple'] += 1

    f.close()

    with open("./wordlist/Oxford 5000.txt", "r") as g:
        for line in g:
            for j in conversation:
                if j+"\n" == line:
                    vocab_complexity['complex'] += 1
    g.close()

    result = json.dumps(vocab_complexity)
    return result


'''
    Function for getting the unique words from the text (using TF-IDF algorithm)
'''


def get_unique_words(conversation):
    data = [i for i in conversation.split()]
    cv = CountVectorizer()
    data = cv.fit_transform(data)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(data)
    word2tfidf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))

    tfidf_array = []
    for word, score in word2tfidf.items():
        tfidf_array.append((word, score))

    return tfidf_array


'''
    Function for getting the number of unique words
'''


def get_number_unique_words(conversation):
    dictionary = {}
    conversation_list = conversation.split()
    unique_words = set(conversation_list)

    return len(unique_words)


'''
    Function for getting the word frequency
'''


def get_word_frequency(conversation):
    dictionary = {}
    conversation_list = conversation.split()
    unique_words = set(conversation_list)

    for words in unique_words:
        dictionary[words] = conversation_list.count(words)

    return dictionary


'''
    Function for getting the conversation topics using LDA algorithm
'''


def get_conversation_topic(conversation):
    conversation = conversation.lower()
    conversation = gensim.parsing.preprocessing.remove_stopwords(conversation)
    conversation = conversation.split()
    conversation_no_stop_words = [word for word in conversation if not word in stopwords.words()]

    conversation = [conversation_no_stop_words]

    dictionary = corpora.Dictionary(conversation)
    document_term_matrix = [dictionary.doc2bow(word) for word in conversation]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(document_term_matrix, num_topics=3, id2word=dictionary, passes=50)

    return ldamodel.print_topics(num_topics=3, num_words=4)


'''
    Get the topics of the conversation
    return: json (list of lists)
'''


@app.route('/getUniqueWords', methods=['POST'])
def getUniqueWords():
    req_json = request.json
    conversation = req_json["Conversation"]
    data = get_unique_words(conversation)

    return jsonify(data)


@app.route('/getConversationTopics', methods=['POST'])
def getTopics():
    req_json = request.json
    conversation = req_json["Conversation"]
    data = get_conversation_topic(conversation)

    return jsonify(data)


'''
    Get the frequency of each word
    return: dict
'''


@app.route('/getWordFrequency', methods=['POST'])
def getWordFrequency():
    req_json = request.json
    conversation = req_json["Conversation"]
    data = get_word_frequency(conversation)

    return jsonify(data)


'''
    Get the number of Unique Words within a corpus
    return: int
'''


@app.route('/getNumberUniqueWords', methods=['POST'])
def getNumberUniqueWords():
    req_json = request.json
    conversation = req_json["Conversation"]
    data = get_number_unique_words(conversation)

    return jsonify(data)


@app.route('/getVocabularyComplexity', methods=['POST'])
def getVocabularyComplexity():
    req_json = request.json
    conversation = req_json["Conversation"]
    data = get_vocabulary_complexity(conversation)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
