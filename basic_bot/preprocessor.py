import spacy
from os import listdir
from os.path import isfile, join
import numpy as np

nlp = spacy.load('en')

data_path = '/home/nirvan/workarea/chatbot/src/FinanceBot/basic_bot/data/intent_classes/'
#data_path = '/data/intent_classes/'


#list of filenames after removing the extensions.
labels = [f.split('.')[0] for f in listdir(data_path) if isfile(join(data_path, f))]

class Dataset(object):
	def __init__(self):
		vocab = nlp.vocab
		X_all_sent = []       #list of all sentences across all files
		
		X_all_vec_seq = []    # list of lists of word vectors. Each element in main list is a list of word vectors of a particular sentence.
		                      # size: (no. of sentences X no. of words in each sentence)
		                       
		X_all_doc_vec = []    #list of vectors of entire sentences
		                      # size: (no. of sentences)
		
		Y_all = []            #list of labels corressponding to all sentences. 
		                      # size: (no. of sentences)
		for label in labels:
			x_file = open(data_path+label + '.txt')
			
			#list of sentences in file
			x_sents = x_file.read().split('\n')
			for x_sent in x_sents:
				if len(x_sent) > 0:
					x_doc = nlp(x_sent)
					x_doc_vec = x_doc.vector	
					x_vec_seq = []
					for word in x_doc:
						x_vec_seq.append(word.vector.astype(np.float64))
					X_all_sent.append(x_sent)
					X_all_doc_vec.append(x_doc_vec)
					X_all_vec_seq.append(x_vec_seq)
					Y_all.append(label)

		self.X_all_sent = X_all_sent
		self.X_all_vec_seq = X_all_vec_seq
		self.X_all_doc_vec = X_all_doc_vec
		self.Y_all = Y_all

def pad_vec_sequences(sequences,maxlen=50):
	new_sequences = []
	for sequence in sequences:
		
		'''
		maxlen is maximum length of sentences. orig_len is no. of words in the sentence, vec_len is length of word vectors.
		'''
		
		orig_len, vec_len = np.shape(sequence)
		if orig_len < maxlen:
			new = np.zeros((maxlen,vec_len))
			new[maxlen-orig_len:,:] = sequence
		else:
			#print(sequence)
			new = sequence[orig_len-maxlen:,:]
		new_sequences.append(new)
	new_sequences = np.array(new_sequences)
	#print(new_sequences.shape)
	return new_sequences

'''
Converts labels array to one-hot vector. 
'''	
def pad_class_sequence(sequence, nb_classes):
	return_sequence = []
	for label in sequence:
		new_seq = [0.0] * nb_classes
		new_seq[labels.index(label)] = 1.0
		return_sequence.append(new_seq)
	return return_sequence	
		
