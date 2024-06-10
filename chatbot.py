import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Load necessary data and model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('my_model.keras')

# Define functions for text preprocessing and intent prediction
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Define a function to retrieve a response based on detected intent
def get_response(intents_list, intents_data):
    tag = intents_list[0]['intent']
    list_of_intents = intents_data['intents']
    
    for intent in list_of_intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)
    
    return "I'm sorry, I don't have a response for that at the moment."

# Define a fallback response in case no intent is detected
def fallback_response():
    fallback_responses = [
        "I'm sorry, I didn't understand that.",
        "Could you please rephrase?",
        "I'm not sure I understand. Can you try again?",
        "Apologies, I didn't get that. Can you provide more details?"
    ]
    return random.choice(fallback_responses)

# Main loop for user interaction
print("GO! Bot is running!")

while True:
    message = input("You: ")
    detected_intents = predict_class(message)
    print("Detected Intents:", detected_intents)  # Debug statement
    
    if detected_intents:
        response = get_response(detected_intents, intents)
        print("Selected Response:", response)  # Debug statement
    else:
        response = fallback_response()
    
    print("Bot:", response)
