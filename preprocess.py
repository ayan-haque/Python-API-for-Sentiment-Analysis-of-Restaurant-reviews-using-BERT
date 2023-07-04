import re

class clean:
	
	def clean_text(self, review):
		corpus = []
		review = re.sub('[^a-zA-Z0-9]', ' ', review)
		review = review.lower()
		review = re.sub(' +', ' ', review)
		review = review.strip()
		corpus.append(review)
		return corpus
	
	def main(self, data):
		processed_data = {}
		for key,value in data.items():
			processed_data[key] = self.clean_text(value)
		return processed_data
