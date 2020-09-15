import json
import pandas as pd
from pycorenlp import StanfordCoreNLP
from main import get_unique_words, get_word_frequency, get_number_unique_words, get_vocabulary_complexity
nlp = StanfordCoreNLP('http://localhost:9000')

df = pd.read_csv("knowledgeGraphData/Output_clean.csv")

comments = []
posDictionary = {}

for i in df["transcript_corrected (S)"]:
    comments.append(i)

for comment in comments:
    print("--- Comment ----")
    print(comment)
    result = nlp.annotate(comment,
                          properties={
                              'annotators': 'ner, pos',
                              'outputFormat': 'json',
                              'timeout': 5000,
                          })

    for j in range(0, len(result['sentences'])):
        posList = []
        for k in range(0, len(result['sentences'][j]['tokens'])):
            pos = result['sentences'][j]['tokens'][k]['pos']
            posList.append(pos)
        posString = str(posList).strip('[]')
        # ADD Pos to the Dictionary of POS
        if posString in posDictionary:
            posDictionary[posString] += 1

        else:
            posDictionary[posString] = 1

    uniqueWords = get_unique_words(comment)
    wordFrequency = get_word_frequency(comment)
    numberUniqueWords = get_number_unique_words(comment)
    vocabularyComplexity = get_vocabulary_complexity(comment)

    print("Word Frequency: ", wordFrequency)
    print("Number of Unique Words: ", numberUniqueWords)
    print("Vocabulary Complexity: ", vocabularyComplexity)
#print(posDictionary)
