import gensim
from flask import Flask, request, jsonify
from gensim import corpora

app = Flask(__name__)


def get_conversation_topic(conversation):
    list = conversation.split()
    dictionary = corpora.Dictionary(list)
    document_term_matrix = [dictionary.doc2bow(word) for word in list]
    Lda = gensim.models.ldamodel.LdaModel

    ldamodel = Lda(document_term_matrix, num_topics=3, id2word=dictionary, passes=50)

    return ldamodel.print_topics(num_topics=5, num_words=3)


@app.route('/getConversationTopics', methods=['POST'])
def getTopics():
    req_json = request.json
    conversation = req_json["Conversation"]
    data = get_conversation_topic(conversation)

    return jsonify(data)


if __name__ == '__nlpServer__':
    app.run(debug=True)
