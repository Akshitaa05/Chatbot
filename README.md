## Overview 
It Is a project based on NLP and Machine learning . In this I have made a chatbot to answer queries like human.

✅What is a Chatbot?
A chatbot is a computer program designed to simulate human conversation. Chatbots can be used for a wide range of purposes, from answering customer service inquiries to providing recommendations for products and services.

## Table of Contents
1. [Setting Up Environment](#setting-up-environment)
2. [Creating Intent Training Data](#creating-intent-training-data)
3. [Installing Libraries](#installing-libraries)
4. [Importing Libraries](#importing-libraries)
5. [Creating Variables](#creating-variables)
6. [Preprocessing Data](#preprocessing-data)
7. [Preparing Training Data](#preparing-training-data)
8. [Building Neural Network Model](#building-neural-network-model)
9. [Compiling and Training Model](#compiling-and-training-model)
10. [Saving the Model](#saving-the-model)
11. [Conclusion](#conclusion)

## Let's discuss the programm step by step.
1. Setting Up Environment <a name="setting-up-environment"></a>
Create a virtual environment using the following command:
```python
python -m venv chatbot
```
Activate the environment using the following command:
```python
  chatbot\Scripts\activate
```
### Benifits of using a virtual environment
1. Isolation of Dependencies
A virtual environment isolates your project's dependencies from the system-wide Python environment. This means you can have specific versions of libraries and packages for your chatbot project without interfering with other projects or the global Python installation.
2. Avoiding Dependency Conflicts
Different projects might require different versions of the same package. A virtual environment allows you to maintain separate versions of packages for each project, preventing conflicts. For example, your chatbot project might need a specific version of rasa, while another project might need a different version.
3. Reproducibility
Using a virtual environment ensures that your project can be set up consistently across different development environments. You can easily share your project with others and ensure that they have the same setup by using a requirements.txt file to list all dependencies.
4. Simplified Package Management
Within a virtual environment, you can install, update, and manage packages specific to your project without requiring administrative privileges. This simplifies the development process and avoids potential issues with system-wide package management.
5. Enhanced Security
By isolating your project's dependencies, you minimize the risk of inadvertently affecting system-wide libraries, which can be critical for maintaining the integrity and security of your system.

2. Creating Intent Training Data<a name="creating-intent-training-data"></a>
 Create a intent.json file on vs code.
 ```python
  {"intents": [
    {"tag": "greeting",
     "patterns": ["Hi there", "How are you", "Is anyone there?", "Hey", "Hola", "Hello", "Good day"],
     "responses": ["Hello", "Good to see you again", "Hi there, how can I help?"],
     "context": [""]
    },
    {"tag": "goodbye",
     "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
     "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
     "context": [""]
    },
    {"tag": "thanks",
     "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
     "responses": ["My pleasure", "You're Welcome"],
     "context": [""]
    },
    {"tag": "query",
     "patterns": ["What is Machine Learning?"],
     "responses": ["Machine learning (ML) is a branch of artificial intelligence (AI) and computer science that focuses on the using data and algorithms to enable AI to imitate the way that humans learn, gradually improving its accuracy. "],
     "context": [""]
    } 
]}
```
This file contains the training data for the chatbot and helps define how the bot should respond to various user inputs.
Contain predefined intents and associated patterns and responses.
Purpose of the intents.json File
1. Define Intents:
 An intent represents the purpose or goal of a user's input. For instance, when a user says "Hi there" or "Hello," they have the intent to greet the chatbot.
 Each intent has a tag (e.g., "greeting", "goodbye") that uniquely identifies it.
2. Patterns:
 Under each intent, there are patterns which are examples of what a user might say to express that intent. For the "greeting" intent, patterns include "Hi there", "Hello", and "Good day".
 These patterns help the chatbot recognize user input and map it to the correct intent.
3. Responses:
 Each intent has a list of possible responses. When the chatbot detects an intent, it will reply with one of these predefined responses.
 For example, when the user greets the bot, the bot might respond with "Hello" or "Hi there, how can I help?".
4. Context:
The context field can be used to manage the conversation state and flow, although it's not used in your example. Contexts allow the bot to keep track of where it is in a conversation and respond appropriately based on previous interactions.

3. Installing Libraries <a name="installing-libraries"></a>
Download the required libarires.
 ```python
  pip install numpy!
  pip install tensorflow 
  pip install nltk
```
1. NumPy is a fundamental package for scientific computing with Python. NumPy is commonly used in machine learning and data science projects for numerical computations.
2. TensorFlow is widely used for developing deep learning models, including neural networks for tasks such as image recognition, natural language processing, and more.
3. Natural Language Toolkit is a leading platform for building Python programs to work with human language data.

4. Importing Libraries<a name="importing-libraries"></a>
Import the libarires in your project using
 ```python
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
```
1. The random module provides functions for generating pseudo-random numbers
2. The json module provides functions for encoding and decoding JSON (JavaScript Object Notation) data.The json module allows you to convert Python data structures (such as dictionaries and lists) to JSON strings, and vice versa.
3. The pickle module provides functions for serializing and deserializing Python objects. Serialization is the process of converting a Python object into a byte stream, which can then be written to a file or transmitted over a network. Deserialization is the process of reconstructing a Python object from a byte stream. Pickle is commonly used for saving and loading complex data structures, such as machine learning models or configuration settings.
4. NumPy is a fundamental package for scientific computing with Python. NumPy is commonly used in machine learning and data science projects for numerical computations.
5. TensorFlow is widely used for developing deep learning models, including neural networks for tasks such as image recognition, natural language processing, and more
6. Natural Language Toolkit is a leading platform for building Python programs to work with human language data.
7. This line specifically imports the WordNetLemmatizer class from the nltk.stem module. Word lemmatization is the process of reducing a word to its base or root form, often used in natural language processing tasks.











