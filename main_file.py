import sys
sys.path.append("<your_custom_path>/Python-API-for-Sentiment-Analysis-of-Restaurant-reviews-using-BERT")
import numpy as np
from numpy import array, amax, amin, sum
import pickle
from scipy.spatial.distance import cdist
from preprocess import clean
clean_obj = clean()
from embeddings import Embedding
Embedding_obj = Embedding()

filename = '<your_custom_path>/Python-API-for-Sentiment-Analysis-of-Restaurant-reviews-using-BERT/ML_model_trained_weights/naive_bayes_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename_2 = '<your_custom_path>/Python-API-for-Sentiment-Analysis-of-Restaurant-reviews-using-BERT/Train_data_BERT_Embeddings/RR_Positive_Train_data_Bert_embeddings.sav'   
filename_3 = '<your_custom_path>/Python-API-for-Sentiment-Analysis-of-Restaurant-reviews-using-BERT/Train_data_BERT_Embeddings/RR_Negative_Train_data_Bert_embeddings.sav'   

RR_Positive_dataset_emd = pickle.load(open(filename_2, 'rb'))  
RR_Negative_dataset_emd = pickle.load(open(filename_3, 'rb'))  


class  Restaurant_Reviews:
	
	def predict(self, df):
		sample_data = df.iloc[0,:]
		print('The shape of Input_sample_data_embd_is: ', np.shape(sample_data))
		sample_data_np  = np.array(sample_data)  
		sample_data_2d  = sample_data_np.reshape(1,-1)
		print('The shape of Sample_data_2d: ', np.shape(sample_data_2d)) 
		cosine_dist_pos = []
		for i_emd_p in RR_Positive_dataset_emd:
			score_matrix_p = cdist(sample_data_2d, i_emd_p, 'cosine')
			score_matrix_p = 1 - score_matrix_p
			cosine_dist_pos.append(score_matrix_p)
		cosine_dist_neg = []
		for i_emd_n in RR_Negative_dataset_emd:
			score_matrix_n = cdist(sample_data_2d, i_emd_n, 'cosine')
			score_matrix_n = 1 - score_matrix_n
			cosine_dist_neg.append(score_matrix_n)
			
		sum_pos = sum(cosine_dist_pos)
		sum_neg = sum(cosine_dist_neg)
		mean_pos = np.mean(cosine_dist_pos)
		mean_neg = np.mean(cosine_dist_neg)
		
		classifier = loaded_model
		result = classifier.predict(sample_data_2d)
		
		result_dict = {}
		if((sum_pos>180 or sum_neg>180) and (mean_pos>0.30 or mean_neg>0.30)):
			if(result == 1):
				result_dict['result'] = 'Wow! its a Positive Review'
			elif(result == 0):
				result_dict['result'] = 'Oops.. its a Negative Review'
		else:
			result_dict['result'] = 'Please write proper reviews, related to your Restaurant experience'
			
		return result_dict
        	
		

	def main(self, request_data):
		processed_data = clean_obj.main(request_data)
		print('processed_data is :', processed_data) 
		for key, value in processed_data.items():
			sentence = processed_data[key]
			print('sentence is :', sentence)
		df = Embedding_obj.get_embeddings(sentence)
		return self.predict(df)
	
	
