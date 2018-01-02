Author: Nirvan Anjirbag


This is a Chatbot that uses deep learning and intent recognition. It is currently trained on a small number of intents. It recognizes the intent of the user's query and gives the appropriate response. 

The chat bot is in python. Text is entered by the user as a command line argument. The python script uses a model which has been trained on many intents. Using this model, it predicts the intent of the user entered text. For each intent there is a single fixed response, which the chat bot gives.


#REQUIREMENTS and SETUP:

I would recommend the following way.

In my case, I disabled anaconda from bashrc and enabled python3.5
Then I installed all libraries using pip (pip3)
Henceforth python refers to python 3.5, so it's best if you alias python3 to python. So then just use python everywhere instead of writing python3.

Requirements:
python3.5
spacy 2.0.5
keras 2.1.2
tensorflow 1.4.1
sys
numpy
scikit-learn
scipy
cython
nltk (I guess it's fine if this is not there)
hdf5

After installing spacy you need to get the spacy model called en. Do:
  
   python -m spacy download en


#Running the code

Once basic_bot is running on its own and everything is installed, proceed as follows. 

#usage
To run this,

1. Train the model on the intents. Go to basic_bot and do `python intent_train.py` . This will train the model on the intents mentioned in `data/intent_classes` . Every text file is of one intent. Every file has a collection of statements on which that intent is trained. The filename is the intent label.

2. Run `python intent_predict.py` . Enter any text as a command line argument. If the text entered is relevant to any of the intents on which it is trained, it will reply with the answer to that intent.  
-----------------------------
Looking at the intent detection code

The intent detection code is located in basic_bot. This can detect intents and works on the terminal. To try this out go to src/FinanceBot/basic-bot and do `python intent_predict.py` . You can enter a statement in the terminal and it will tell you the intent. 

I have created a new file intent_predict2.py from intent_predict.py . The difference is that intent_predict2.py receives the text from the terminal and then prints the response corresponding to the detected intent. 
-------------------------------
Adding more intents

The intent data is located in basic_bot/data/intent_classes. Each text file is one intent. The name of each file becomes the official intent name. The statements in each file are used to train the model on that intent. So to add a new intent simply create a new text file here. Name it after the intent. Add statements to it on which you want the model to be trained. After adding this file, retrain the model using #1 mentioned above. The chatbot can now detect that intent. But you also need to add a response message for that intent. Go to basic_bot/intent_predict2.py. Add your response string with the other responses. Go to def process_labels() and add an if statement, for your new intent. The chatbot will now return that message when it detects this intent.


