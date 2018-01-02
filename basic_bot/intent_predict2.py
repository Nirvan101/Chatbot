try:
       
        #print('Before keras import')
        #import logging
        #logging.getLogger("keras").setLevel(logging.WARNING)
        #import sys
        #stdout = sys.stdout
        #sys.stdout=open('/dev/null','w')
        #print(sys.path)
        #import tensorflow as tf
        #print(tf.__version__)
        #import keras
        #sys.stdout=stdout
        #print(keras.__version__)
        
        import warnings
        warnings.filterwarnings("ignore")
        
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        from keras.models import load_model
        from preprocessor import nlp
        
        from preprocessor import pad_vec_sequences, labels
        import spacy
        import numpy as np
        import sys
        from dependency_tree import to_nltk_tree , to_spacy_desc

except Exception as inst:
        a = 6
        print('Exception occured')
        print(inst)
        
#print(labels)

#-----------------------------------------------------------
#responses to the intents

greetings_ans = 'Hello'

thanks_ans = 'Welcome'


about_dl_ans =  'Deep Learning is a class of machine learning algorithms that use multiple layers to extract features. Higher level features are extracted from lower level features to understand the input. Applications of Deep Learning include - Automatic Speech Recognition, Image Processing, Natural Language Processing and many more.'

about_events_ans = 'We keep on organizing Faculty Development Programs and Workshops in colleges. We also organize internal knowledge sharing sessions.'


about_office_timimgs_ans = 'We work from 10 am to 6 pm'

about_weather_ans = "I'll have to check with the weather man"

#------------------------------------------------------------

nb_classes = len(labels)

#load the model to be tested.
model = load_model('/home/nirvan/workarea/chatbot/src/FinanceBot/basic_bot/backup/intent_models/model2.h5')

n_test = 1
test_vec_seq = [] #list of all vectorized test queries
test_ent_seq = [] #list of lists of entities in each test query
test_seq = [] #list of all test queries
for i in range(n_test):
	test_text = sys.argv[1]
	test_seq.append(test_text)
	#vectorize text.
	test_doc = nlp(test_text)
	test_vec = []
	for word in test_doc:
		test_vec.append(word.vector)
	test_vec_seq.append(test_vec)
	test_ent_seq.append(test_doc.ents)
	
#convert all the sentences into matrices of equal size.
test_vec_seq = pad_vec_sequences(test_vec_seq)
#get predictions
prediction = model.predict(test_vec_seq)

label_predictions = np.zeros(prediction.shape)

for i in range(n_test):
	m = max(prediction[i])
	p = np.where(prediction[i] > 0.55 * m)	# p collects possible sub intents
	q = np.where(prediction[i] == m)	#q collects intent
	label_predictions[i][p] = 1
	label_predictions[i][q] = 2

def process_labels(label):
	if(label == 'greetings'):
		return greetings_ans
	if(label == 'thanks'):
		return thanks_ans
	if(label == 'about_dl'):
	        return about_dl_ans
	if(label == 'about_event'):
	        return about_events_ans
	if(label == 'about_office_time'):
	        return about_office_timimgs_ans
	if(label == 'about_weather_ans'):
	        return about_office_timimgs_ans
	return 'Could not detect intent'

for i in range(n_test):
	for x in range(len(label_predictions[i])):
		if label_predictions[i][x] == 2 :
			print(process_labels(labels[x]))
			#print(" Detected intent: ",labels[x])
	
