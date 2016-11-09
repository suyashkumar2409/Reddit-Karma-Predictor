import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import string
import re
import nltk
import enchant
import sklearn
import pickle


class InputClass:
	category = None
	author_comment_karma = None
	author_link_karma = None
	created_utc = None
	num_comments = None
	thumbnail = None
	type_var = None
	author_is_gold = None
	title = None
	ups = None


class Prediction:
	class1 = None
	class2 = None
	test_text = None
	test_df = None
	train_df = None
	final_train_df = None
	hashingVect = None
	text_train = None
	final_attr = None
	ups_factor = None
	partial_clean = None

	def __init__(self):
		self.class1 = pickleHandle.load_object('class1.pkl')
		self.class2 = pickleHandle.load_object('class2.pkl')
		self.train_df = pd.read_pickle('IR-data.p')
		self.final_train_df = pd.read_pickle('PostClass2.p')
		self.hashingVect = pickleHandle.load_object('hashingVect.pkl')
		self.final_train_df = pd.read_pickle('PostClass2.p')
		self.test_df = self.final_train_df.loc[0,:].copy(deep = True)
		self.ups_factor = 38331
		self.partial_clean = pd.read_pickle('CleanedData_2.p')


	def one_hot_conv(self, category,string):
		#print(self.train_df)
		lis = self.partial_clean[string].unique()
		if string == 'thumbnail':
			np.append(lis,"link")
			#print(lis)
		
		if string == 'type':
			np.append(lis,"txt")



		for col in lis:
			if col == category:
				self.test_df[string + "_" + str(col)] = 1
			else:
				self.test_df[string + "_" + str(col)] = 0

	def numerical_conv(self, inputc):
		attr = ['created_utc','author_link_karma','author_comment_karma', 'num_comments']

		avg = self.train_df["created_utc"].mean()
		diff = self.train_df["created_utc"].max() - self.train_df["created_utc"].min()
		self.test_df["created_utc"] = (inputc.created_utc - avg)/diff - self.final_train_df["created_utc"].min()
		
		avg = self.train_df["author_link_karma"].mean()
		diff = self.train_df["author_link_karma"].max() - self.train_df["author_link_karma"].min()
		self.test_df["author_link_karma"] = (inputc.author_link_karma - avg)/diff - self.final_train_df["author_link_karma"].min()
        
		avg = self.train_df["author_comment_karma"].mean()
		diff = self.train_df["author_comment_karma"].max() - self.train_df["author_comment_karma"].min()
		self.test_df["author_comment_karma"] = (inputc.author_comment_karma - avg)/diff - self.final_train_df["author_comment_karma"].min()

		avg = self.train_df["num_comments"].mean()
		diff = self.train_df["num_comments"].max() - self.train_df["num_comments"].min()
		self.test_df["num_comments"] = (inputc.num_comments - avg)/diff - self.final_train_df["num_comments"].min()

	def string_conv(self, inputc):	
		temp = str(textHandle.cleanstring(inputc.title))
		temp = str(textHandle.removenonsensewords(temp))
		bla = str(textHandle.removebadwords(temp,textHandle.listofbadwords()))
		temp = str(textHandle.replacewithstem(temp))
		#print(temp)
		self.text_train = self.hashingVect.transform(temp)



	def transform_data(self, inputc):
		one_hot_lis = ['Category','author_is_gold','thumbnail','type']

		self.one_hot_conv(inputc.category, 'Category')
		self.one_hot_conv(inputc.author_is_gold, 'author_is_gold')
		self.one_hot_conv(inputc.thumbnail, 'thumbnail')
		self.one_hot_conv(inputc.type_var, 'type')

		self.numerical_conv(inputc)
		self.string_conv(inputc)

		self.test_df['predicted_ups'] = self.textPrediction()

	def textPrediction(self):
		#print(self.text_train.shape)
		#print(self.class1.predict(self.text_train))

		return self.class1.predict(self.text_train)[0]

	def finalPrediction(self):
		self.final_attr = [ u'thumbnail_default',        u'thumbnail_link',
              u'thumbnail_nsfw',        u'thumbnail_self',
                    u'type_img',              u'type_txt',
                    u'type_vid',  u'author_is_gold_False',
         u'author_is_gold_True',      u'Category_Android',
                u'Category_Art',    u'Category_AskReddit',
            u'Category_Bitcoin', u'Category_Conservative',
                u'Category_DIY',        u'Category_Jokes',
                u'Category_MMA',        u'Category_Music',
                u'Category_WTF',        u'Category_apple',
         u'Category_askscience',       u'Category_bestof',
               u'Category_cats',       u'Category_comics',
             u'Category_creepy',        u'Category_drunk',
           u'Category_facepalm',         u'Category_food',
              u'Category_funny',         u'Category_guns',
          u'Category_lifehacks',       u'Category_movies',
                u'Category_nba',         u'Category_news',
           u'Category_politics',  u'Category_programming',
               u'Category_rage',      u'Category_science',
                u'Category_sex',       u'Category_soccer',
              u'Category_space',   u'Category_technology',
          u'Category_teenagers',        u'Category_trees',
             u'Category_trippy',       u'Category_videos',
          u'Category_worldnews',  
        u'author_comment_karma',     u'author_link_karma',
                 u'created_utc',          u'num_comments',
               u'predicted_ups']

		#print(self.class2.predict(self.test_df[self.final_attr]))
		print(self.test_df[self.final_attr])
		return self.class2.predict(self.test_df[self.final_attr]) * self.ups_factor

class pickleHandle:
	@staticmethod
	def save_object(obj, filename):
	    with open(filename, 'wb') as output:
	        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load_object(filename):
	    pkl_file = open(filename, 'rb')
	    mydict2 = pickle.load(pkl_file)
	    pkl_file.close()
	    return mydict2

class textHandle:
	@staticmethod
	def removepunctuation(x):
	    x = x.replace('.','')
	    x = x.replace(')','')
	    x = x.replace('(','')
	    x = x.replace('\'','')
	    x = x.replace('-','')
	    x = x.replace('?','')
	    x = x.replace('$','')
	    x = x.replace('!','')
	    x = x.replace(':','')
	    x = x.replace(',','')
	    x = x.replace('%','')
	    x = x.replace('+','')
	    x = x.replace('*','')
	    x = x.replace('/','')
	    x = x.replace('&','')
	    x = x.replace('@','')
	    #replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
	    #x = str(x).translate(replace_punctuation)
	    #retstr = x.translate(string.maketrans("",""), string.punctuation)
	    return x
	    
	@staticmethod
	def removeunicode(x):
	    return re.sub(r'[^\x00-\x7F]+',' ', x)
	@staticmethod
	def lowercasestring(x):
	    return x.lower()

	@staticmethod
	def removedigits(s):
	    s = re.sub(" \d+", " ", s)
	    return s
	    
	@staticmethod
	def cleanstring(x):
	    #x=replaceredundancy(x)
	    x=textHandle.removepunctuation(x)
	    x=textHandle.removeunicode(x)
	    #x = trimstring(x)
	    x=textHandle.removedigits(x)
	    x=textHandle.lowercasestring(x)
	    return x 

	@staticmethod
	def removenonsensewords(text):
		d = enchant.Dict("en_US")
		tokens = nltk.word_tokenize(text)
		stemmed = []
		for token in tokens:
			if d.check(token):
				stemmed.append(token)

		return ' '.join(stemmed)

	@staticmethod
	def listofbadwords():
	    from nltk.corpus import stopwords
	    stopwords = stopwords.words('english')
	    monthnames = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
	    randomrepetitive = ['edu','unl','mt']
	    
	    totlist = stopwords + monthnames + randomrepetitive
	    return totlist
	@staticmethod
	def removebadwords(x,totlist):
    
	    wordlist = x.split()
	    wordlist = [word for word in wordlist if word.lower() not in totlist]
	    x = ' '.join(wordlist)
	    return x

	@staticmethod
	def replacewithstem(text):
	    tokens = nltk.word_tokenize(text)
	    stemmer = nltk.stem.porter.PorterStemmer()
	    
	    stemmed = []
	    for token in tokens:
	        stemmed.append(stemmer.stem(token))
	        
	    return ' '.join(stemmed)

def main():
	inputc = InputClass()
	inputc.category = 'AskReddit'
	inputc.author_comment_karma = 270316.0
	inputc.author_link_karma = 11657.0
	inputc.created_utc = 100000000
	inputc.num_comments =  	7769
	inputc.thumbnail = 'link'
	inputc.type_var = 'txt'
	inputc.author_is_gold = True
	inputc.title = "What tasty food would be distusting if eaten over rice?"


	pred = Prediction()
	pred.transform_data(inputc)
	print(int(round(pred.finalPrediction()[0])))

main()