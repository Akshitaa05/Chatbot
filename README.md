![image](https://github.com/Akshitaa05/Chatbot/assets/171258488/6cb49af8-d5ba-4e25-970a-f4d7559630f4)# Chatbot

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


