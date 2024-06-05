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
#### 1. Setting Up Environment <a name="setting-up-environment"></a>
Create a virtual environment using the following command:
```python
python -m venv chatbot
```
Activate the environment using the following command:
```python
  chatbot\Scripts\activate
```
#### Benifits of using a virtual environment
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

#### 2. Creating Intent Training Data<a name="creating-intent-training-data"></a>
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

#### 3. Installing Libraries <a name="installing-libraries"></a>
Download the required libarires.
 ```python
  pip install numpy!
  pip install tensorflow 
  pip install nltk
```
1. NumPy is a fundamental package for scientific computing with Python. NumPy is commonly used in machine learning and data science projects for numerical computations.
2. TensorFlow is widely used for developing deep learning models, including neural networks for tasks such as image recognition, natural language processing, and more.
3. Natural Language Toolkit is a leading platform for building Python programs to work with human language data.

#### 4. Importing Libraries<a name="importing-libraries"></a>
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

#### 5. Creating Variables<a name="creating-variables"></a>
Create variables for further use using
 ```python
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']
```
1. The line lemmatizer = WordNetLemmatizer() creates an instance of the WordNetLemmatizer class from the nltk.stem module. This instance, often referred to as lemmatizer, allows you to perform lemmatization on words in your text data.
2. The line intents = json.loads(open('intents.json').read()) is used to load JSON data from a file named 'intents.json' into a Python dictionary.
open('intents.json').read(): This part of the line opens the file named 'intents.json' in the current directory and reads its contents as a string.The json.loads() function is used to parse JSON data from a string into a Python dictionary.
###### Wordnetlematizer is used to reduce the words to its base form(EG:- running,ran,runned all converted to its root form run.) to improve accuracy.
3. The word list is typically used to store all the words found in the training data. Each word will represent a unique token in the text data. The words will be extracted from the patterns in the training data, and duplicates will be removed to create a vocabulary.
4. The classes list is used to store all the unique classes or intents present in the training data. Each class represents a category or label that the chatbot should be able to recognize and respond to.
5. The document list is used to store tuples containing the patterns (inputs) and their corresponding class or intent labels (outputs). Each tuple represents a training example. For example, if a pattern like "Hi there" is associated with the intent "greeting", it will be stored as a tuple ("Hi there", "greeting") in the documents list.
6. The ignoreletters list contains characters that should be ignored or removed from the patterns during preprocessing.

#### 6. Preprocessing the Training Data<a name="preprocessing-data"></a>
Now, we will preprocess the data and make it suitable for usage.
 ```python
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
```
### Let's Learn the  functioning of each line:-
1. In the first for loop, we traverse through each intent (column/different types of segments) of the intents file. In intent we have patterns so the inner for loop is for traversing through each pattern one by one.
A pattern is a set of words user might give to the chatbot . Wordlist is a list that creates tokens of each pattern . We tokenise each pattern and store the tokens in wordlist(a list).
Now each token is added to the empty word list we created to create the vocabulary. of the chatbot.
Next we add the tuple containing each token and the tag it belongs to the document list(empty list we created).
Now if the tag is already present good. But if not add it to the classes list.(stores unique tags).
    ###### This segment of code is crucial for preprocessing the training data and organizing it into a format suitable for training a machine learning model. It extracts words from patterns, builds a vocabulary, creates training examples, and identifies unique intent classes, laying the groundwork for training the chatbot model to recognize and respond to user inputs effectively.Prepared the data for training.
2. In the next few lines of code, a new list words is created which contains each instance in the word list lemmatized (converted to its root form) only in case if it is not in ignoreletters list*(actual words).
Then the list  words is converted to set(to remove duplicate items) and is sorted alphabetically to form a proper vocabulary.
Similarly classes list is converted to set(to remove duplicate items) and is sorted alphabetically to form a proper vocabulary.
3. In the next part pickle module is introduced.
    #### Pickle Module
    The pickle module in Python provides a way to serialize and deserialize Python objects. Serialization (or "pickling") is the process of converting a Python object into a byte stream, and deserialization (or "unpickling") is the reverse process.
    
    Why Save These Lists?
    	• Reusability: By saving the words and classes lists to files, you can reuse them later without needing to preprocess the raw data again. This can save time, especially if preprocessing is computationally expensive.
    	• Consistency: Ensuring that the same words and classes are used consistently across different runs of your program helps maintain the integrity of your machine learning model. This is crucial for tasks such as feature extraction, where the vocabulary (words) needs to be consistent.
    	• Portability: Serialized files can be shared or transferred between different environments or systems, allowing you to use the same preprocessed data elsewhere.

The lines of code involving pickle.dump are used to save the words and classes lists to disk. This process is called serialization, and it allows you to save Python objects to a file so they can be loaded later without needing to recreate them.

• words: This is the list of unique, lemmatized, and sorted words that was created in the previous steps.
open('words.pkl', 'wb'): This part opens a file named words.pkl in write-binary mode ('wb'). If the file does not exist, it will be created.
pickle.dump(words, ...): This function call serializes the words list and writes it to the words.pkl

• classes: This is the list of unique intent tags that was created during the preprocessing steps.
open('classes.pkl', 'wb'): This part opens a file named classes.pkl in write-binary mode ('wb'). If the file does not exist, it will be created.
pickle.dump(classes, ...): This function call serializes the classes list and writes it to the classes.pkl file.

#### 7.Preparing Training Data<a name="preparing-training-data"></a>
Now, we will train the model by using
 ```python
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]
```
This block of code prepares the training data for a machine learning model by creating a bag-of-words representation for each document and encoding the corresponding intent labels.
Convert each pattern into bag of words(a list of binary values indicating whether each word in words is present or not in the pattern ). The corresponding intent is also converted into one hot encoded vector indicating which intent it belongs to. These bag of words and encoded vectors are combined and added to a list called training.
The training list is shuffled randomly to ensure that neural network does not learn any older dependencies.
Finally the training list is converted into numpy array and split into separate arrays from the bow(trainX) and ev(trainY).

1. training: This list will store the training data, where each element represents a training example.
2. outputEmpty: This list is used as a template for encoding the intent labels(classes/unique list of tags). It contains zeros and has the same length as the classes list. The outputEmpty list is created with all elements initialized to zero. Its length is equal to the number of classes. Each element in outputEmpty corresponds to a class label.
3. For each document in the dataset, let's say we have a document like this: (["hello", "world"], "greeting"), a bag-of-words representation is created.
4. The wordPatterns variable contains the tokenized pattern of the document.
5. Each word in wordPatterns is lemmatized and converted to lowercase.
6. For each word in the words list (which contains the unique vocabulary), a binary value is added to the bag list: 1 if the word is present in the document's word patterns, and 0 otherwise. This creates a binary vector representing which words from the vocabulary are present in the document.
7. An empty list (outputRow) is initialized with zeros using outputEmpty as a template.
8. The index corresponding to the intent label of the document is set to 1, encoding the intent label as a one-hot vector. This means that the position corresponding to the class of the document is set to 1, while all other positions remain 0.
9. The bag-of-words representation (bag) and the one-hot encoded intent label (outputRow) are concatenated and added as a single training example to the training list.
10. The training data is shuffled to ensure that the examples are presented in random order during training.
11. Finally, the training list is converted to a NumPy array for efficient processing and manipulation.
After this process, trainX will contain the input features (bag-of-words representations), and trainY will contain the corresponding output labels (intent labels encoded as one-hot vectors). These arrays can then be used to train a machine learning model.
12. This code block is splitting the training data array into input features (trainX) and output labels (trainY). 
• Explanation of trainX:
	trainX is assigned a portion of the training array.
	training[:, :len(words)] selects all rows (:) of the training array and the first len(words) columns.
	These first len(words) columns represent the input features (bag-of-words representations) of the training data.
• Explanation of trainY:
	trainY is assigned another portion of the training array.
	training[:, len(words):] selects all rows (:) of the training array and all columns starting from the column at index len(words) (exclusive).
	These columns starting from index len(words) represent the output labels (encoded intent labels) of the training data.
	
After this we have set a bag of words for the representation of documents by setting the value of each word in words as 1 if it appears in the list and 0 otherwise.

#### 7. Building Neural Network Model<a name="building-neural-network-model"></a>
Now, we will create a new sequential keras model which is a linear stack of layers. 
Neural network is defined using the tensorflow sequential API which allows the layers to be added in the sequence .Use
 ```python
model = tf.keras.Sequential()
```
This initializes a sequential model in TensorFlow's Keras API. 
A sequential model is a linear stack of layers where each layer has exactly one input tensor and one output tensor. This type of model is suitable for building simple feedforward neural networks.

Next, We add a densely connected neural network layer to the model.
The choice of the number of dense layers in a neural network architecture depends on various factors, including the complexity of the problem, the size of the dataset, and computational resources. Adding more or fewer layers can impact the model's capacity to learn and its performance. 
 ```python
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))
```
This neural network consist of 3 fully connected layers. 
The first two are the input layers which also include the dropout regularization which helps to prevent overfitting.

1. First we add a dense layer with 128 units. input_shape=(len(trainX[0]),): Specifies the input shape for the first layer. len(trainX[0]) corresponds to the number of features in the input data. activation='relu': Applies the ReLU (Rectified Linear Unit) activation function to introduce non-linearity.
2. Then add a dropout layer with a dropout rate of 0.5. Dropout is a regularization technique that randomly sets a fraction of input units to zero during training to prevent overfitting.
3. Then add another dense layer with 64 units. Again, uses the ReLU activation function for introducing non-linearity.
4. After that add another dropout layer with a dropout rate of 0.5.
5. At last add the output layer with units equal to the number of classes in the output (len(trainY[0])). Uses the softmax activation function to produce a probability distribution over the classes.

In summary, the provided code defines a sequential neural network model with two dense layers, each followed by a dropout layer for regularization, and an output layer with softmax activation for multiclass classification.

Dense Layer (Fully Connected Layer):
	1. The dense layer is the fundamental building block of a neural network.
	2. Each neuron in a dense layer is connected to every neuron in the previous layer.
	3. The number of neurons in the layer defines the dimensionality of the output space.
	4. Activation functions, such as ReLU (Rectified Linear Unit), introduce non-linearity to the model, allowing it to learn complex patterns in the data.
Dropout Layer:
	1. Dropout is a regularization technique used to prevent overfitting in neural networks.
	2. During training, a dropout layer randomly sets a fraction of input units to zero.
	3. This helps to prevent the model from relying too heavily on specific features or neurons, forcing it to learn more robust representations.

Let's put it in the context of our model:
1. The first dense layer (Dense(128, activation='relu')) takes the input data and applies a linear transformation followed by the ReLU activation function, resulting in a higher-level representation of the data.(128)
2. The dropout layer (Dropout(0.5)) randomly sets 50% of the input units to zero during training, preventing overfitting by introducing redundancy and reducing the risk of relying too heavily on specific features.(50%*128=64 destroyed).
3. The second dense layer (Dense(64, activation='relu')) further transforms the data, potentially extracting more complex features.(128-64=64 remaining)
4. Another dropout layer is added for regularization.(50%*64=32 destroyed)
5. Finally, the output layer (Dense(len(trainY[0]), activation='softmax')) produces the final predictions, with the softmax activation function converting the model's raw output into probabilities for each class.

#### The final output layer will have the same number of units as the number of classes in the output (len(trainY[0])). The number of units in the hidden layers (128 and 64) does not affect the number of units in the output layer. The dropout layers are used for regularization and do not change the number of units in the network. Therefore, we don't end up with just 32 units remaining in the output layer.



















