# Importing packages
import difflib
import RAKE as rake
import operator
import nltk
import os
import json
import numpy as np
from itertools import chain
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import requests
from nltk.stem.wordnet import WordNetLemmatizer
lmztr = WordNetLemmatizer()
import re
from collections import Counter
import json
import enchant
from enchant import DictWithPWL, request_pwl_dict
from enchant import checker
from flask import Flask, request

app = Flask(__name__)

# List of words for bag of words
words = []

# Classes for classifying action
classes = []

# List of apps name
app_name = ['alarm','alerts','battery','browser','calculator','calendar','call_log','camera','clock','contacts','facebook ','fm_radio','gallery','language','mann_ki_baat','music','nm_app','settings','sos','torch','video','volume','wifi','youtube']

# App features
alarm = ['alarm','clock','stopwatch','timer']
alerts = ['alerts']
battery=['battery']
browser = ['net','browser','internet','google']
calculator = ['calculator','calculate']
calendar = ['calendar','date','month','year']
call = ['name','number','phone','call','ring','phone','dial']
call_log = ['call logs','calllog','calllogs','call log']
camera = ['camera','click pictures','selfie','click pics','click photographs']
clock=['clock', 'alarm', 'stopwatch', 'timer']
contacts = ['phonebook','addressbook','contacts','contact','phone book','address book']
facebook = ['facebook','fb']
fm_radio = ['fm','radio','radio channel','fmradio','fm radio']
gallery = ['photos','pics','pic','gallery', 'photo','pictures','photograph','picture','photographs']
settings = ['settings','setting']
sms = ['message','sms', 'messages', 'messenger','text']
sos = ['emergency call','sos message']
torch = ['on torch','off torch']
video = ['video','videos','video player','videos player','vdo player']
video_call = ['video call','videocall']
volume = ['increase','decrease','volume']
wifi = ['on wifi','off wifi','wi-fi','wifi','wi fi','wai fai','vifi','vai fai']
youtube = ['youtube']

# Spell checking
def rectification(user_input):

    # to find errors
    dict_1 = enchant.DictWithPWL("en_US")
    check = enchant.checker.SpellChecker(dict_1)
    check.set_text(user_input)

    # to find suggestion
    error_input = check.get_text()
    dict_2 = enchant.request_pwl_dict("./sample1.txt")
    for err in check:
        best_words = []
        best_ratio = 0
        sug = set(dict_2.suggest(err.word))
        for b in sug:
            tmp = difflib.SequenceMatcher(None, err.word, b).ratio()
            if tmp > best_ratio:
                best_words = [b]
                best_ratio = tmp
            elif tmp == best_ratio:
                best_words.append(b)

        if len(sug) != 0 and best_ratio > 0.8:
            err.replace(best_words[0])
            # else:
            #     sug = err.suggest()
            #     for b in sug:
            #         tmp = difflib.SequenceMatcher(None, err.word, b).ratio()
            #         if tmp > best_ratio:
            #             best_words = [b]
            #             best_ratio = tmp
            #         elif tmp == best_ratio:
            #             best_words.append(b)
            #     if len(sug) != 0 and best_ratio > 0.8:
            #         err.replace(sug[0])

    rectify_input = check.get_text() # Returns corrected text
    return rectify_input

# Compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1 + np.exp(-x))
    return output

# Convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

# Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, feature_list, show_details = False): #bag of words(bow)
    # Tokenize the pattern
    ignore_words = set(stopwords.words('english'))
    sentence_words = nltk.word_tokenize(sentence)
    words = [w for w in sentence_words if w not in ignore_words]
    bag = [0]*len(feature_list)
    app_detected = list(set(words) & set(app_name))
    if len(app_detected) > 0:
        index = feature_list.index(app_detected[0])
        bag[index] = 1
        app = globals()[app_detected[0]]
        for feature in app:
            index = feature_list.index(feature)
            bag[index] = 1

    # List of synonyms
    synonyms = []
    for w in words:
        # synonyms_list = wordnet.synsets(w)
        # synonyms_list = list(set(chain.from_iterable([word.lemma_names() for word in synonyms_list])))
        # synonyms.extend(synonyms_list)
        synonyms.append(w)
    for word in synonyms:
        if word in feature_list:
            index = feature_list.index(word)
            bag[index] = 1
    #print("Bag of words: ",bag)
    return np.array(bag)

# Calculating the product of input with layer weights
def think(sentence, show_details = False):

    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

    x = bow(sentence.lower(), feature_list, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # Input layer is our bag of words
    layer_0 = x
    # Matrix multiplication of input and hidden layer
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    # Output layer
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    return layer_2 #  Returns weights of layer_2 and list of nouns

# Probability threshold
ERROR_THRESHOLD = 0.5

# Load our calculated synapse values
synapse_file = './synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    feature_list = synapse['features']

# Classifying input sentence into class
def classify(sentence, show_details = False):
    print("sentence")
    classes = synapse['classes']
    results = think(sentence, show_details)
    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key = lambda x: x[1], reverse = True)
    return_results = [[classes[r[0]],r[1]] for r in results]


    if len(return_results) > 0:
        return_results = (return_results[0][0]).split("-")
        action = return_results[0]
        intent = return_results[1]
        if action == 'sms':
            if 'sms' or 'message' or 'text' or 'messages' in sentence:
                action = 'sms'
            else:
                action = 'browser'
    else:
        return_results = ''
        action = 'browser'
        intent = ''

    # Resolved Query processing
    rake_object = rake.Rake("SmartStoplist.txt")
    keywords = rake_object.run(sentence)
    #print("keywords: ",keywords)
    if len(keywords) > 0:
        resolvedQ = keywords[0][0]
    else:
        resolvedQ = ""
    if action == 'browser':
        resolvedQ = ""
        for keyword in keywords:
            resolvedQ += " " + keyword[0]
    if (action == "call" or action == "sms" or action == "video_call") and (len(keywords) > 1) :
        resolvedQ = ""


    processed_data = dict({"post_data":{"action":action,"intent":intent,"resolved_query":resolvedQ}})
    print(processed_data)

    return processed_data


# To get input from server
@app.route('/')
def hello_world():
    user_input = request.args.get('sentence')
    sentence1 = rectification(user_input.lower())
    for item in feature_list:
        if item in sentence1:
            sentence1 = sentence1.replace(item,item.replace(" ",""))

    return json.dumps(classify(sentence1, show_details = True))


if __name__ == "__main__":
    app.run(debug=True)

data_file.close()