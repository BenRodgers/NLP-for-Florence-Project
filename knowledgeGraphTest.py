import json
import pandas as pd
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

df = pd.read_csv("knowledgeGraphData/Output_clean.csv")

comments = []
posDictionary = {}

for i in df["transcript_corrected (S)"]:
    comments.append(i)

for comment in comments:

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
        # ADD Pos to the 
        if posString in posDictionary:
            posDictionary[posString] += 1

        else:
            posDictionary[posString] = 1

print(posDictionary)
