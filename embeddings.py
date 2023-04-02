# This file is for getting the BERT embeddings of 'input_list' containg single 'text'

import pandas as pd
import sys
#sys.path.append("<your_custom_path>/Sentiment-Analysis-of-Restaurant-Reviews")

from sentence_transformers import SentenceTransformer

# MODEL can be downloaded and stored at desired location
# MODEL = SentenceTransformer('bert-base-nli-mean-tokens')

# Sice we've already downloaded the 'bert-base-nli-mean-tokens' model we can use it directly from local system

class Embedding:
	
	def __init__(self):
		#self.MODEL = SentenceTransformer('bert-base-nli-mean-tokens')
		self.MODEL = SentenceTransformer('<your_custom_path>/Sentiment-Analysis-of-Restaurant-Reviews/BERT_models/bert-base-nli-mean-tokens')		
		
		
	def get_embeddings(self, sentence):
    
		# Here sentence is a list # sentence = ["It was okay"]
		sentence_emdeddings = self.MODEL.encode(sentence)
		
		# Creating a 2D embedding matrix for a single sample present in sentence_emdeddings
		df_embedding  = pd.DataFrame(data = sentence_emdeddings)
		return df_embedding
		
	
	
	
	
	
	
		
