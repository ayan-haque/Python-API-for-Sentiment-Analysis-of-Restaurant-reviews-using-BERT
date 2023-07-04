import pandas as pd
import sys
from sentence_transformers import SentenceTransformer

class Embedding:
	
	def __init__(self):
		self.MODEL = SentenceTransformer('<your_custom_path>/Sentiment-Analysis-of-Restaurant-Reviews/BERT_models/bert-base-nli-mean-tokens')		
		
	def get_embeddings(self, sentence):
		sentence_emdeddings = self.MODEL.encode(sentence)
		df_embedding  = pd.DataFrame(data = sentence_emdeddings)
		return df_embedding
		
	
