import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import string
import re
import nltk
import enchant
import sklearn



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

	def __init__(self):
		class1 = pickleHandle.load_object('class1.pkl')
		class2 = pickleHandle.load_object('class2.pkl')
		train_df = pd.read_pickle('IR-data.p')
		final_train_df = pd.read_pickle('PostClass2.p')
		hashingVect = pickleHandle.load_object('hashingVect.pkl')
		test_df = pd.DataFrame()
		ups_factor = 38331


	def one_hot_conv(self, category,string):
		lis = train_df[string].unique()

		for col in lis:
			if col == category:
				self.test_df[string + "_" + col] = 1
			else:
				self.test_df[string + "_" + col] = 0

	def numerical_conv(self, inputc):
		attr = ['created_utc','title','ups','author_link_karma','author_comment_karma']

		avg = train_df[created_utc].mean()
		diff = train_df[created_utc].max() - train_df[created_utc].min()
		self.test_df[created_utc] = (inputc.created_utc - avg)/diff - final_train_df[created_utc].min()
		avg = train_df[title].mean()
		diff = train_df[title].max() - train_df[title].min()
		self.test_df[title] = (inputc.title - avg)/diff - final_train_df[title].min()
        
		avg = train_df[author_link_karma].mean()
		diff = train_df[author_link_karma].max() - train_df[author_link_karma].min()
		self.test_df[author_link_karma] = (inputc.author_link_karma - avg)/diff - final_train_df[author_link_karma].min()
        
		avg = train_df[author_comment_karma].mean()
		diff = train_df[author_comment_karma].max() - train_df[author_comment_karma].min()
		self.test_df[author_comment_karma] = (inputc.author_comment_karma - avg)/diff - final_train_df[author_comment_karma].min()

	def string_conv(self, inputc):	
		self.test_df['title_clean'] = textHandle.cleanstring(inputc.title)
		self.test_df['title_nonsense'] = textHandle.removenonsensewords(self.test_df['title_clean'])
		self.test_df['title_badwords'] = textHandle.removebadwords(self.test_df['title_nonsense'], textHandle.listofbadwords())
		self.test_df['title_stemmed'] = textHandle.replacewithstem(self.test_df['title_badwords'])
		self.text_train = self.hashingVect.transform(self.test_df['title_stemmed'])



	def transform_data(self, inputc):
		one_hot_lis = ['Category','author_is_gold','thumbnail','type']

		self.one_hot_conv(inputc.category, 'Category')
		self.one_hot_conv(inputc.author_is_gold, 'author_is_gold')
		self.one_hot_conv(inputc.thumbnail, 'thumbnail')
		self.one_hot_conv(inputc.type_var, 'type')

		self.numerical_conv(inputc)
		self.string_conv(inputc)

		self.test_df['predicted_ups'] = textPrediction()

	def textPrediction(self):
		return self.class1.predict(self.text_train)

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

		return self.class2.predict(self.test_df[final_attr]) * self.ups_factor


class pickleHandle:
	def save_object(obj, filename):
	    with open(filename, 'wb') as output:
	        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

	def load_object(filename):
	    pkl_file = open(filename, 'rb')
	    mydict2 = pickle.load(pkl_file)
	    pkl_file.close()
	    return mydict2

class textHandle:
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
	    
	def removeunicode(x):
	    return re.sub(r'[^\x00-\x7F]+',' ', x)
	def lowercasestring(x):
	    return x.lower()

	def removedigits(s):
	    s = re.sub(" \d+", " ", s)
	    return s
	    
	def cleanstring(x):
	    #x=replaceredundancy(x)
	    x=removepunctuation(x)
	    x=removeunicode(x)
	    #x = trimstring(x)
	    x=removedigits(x)
	    x=lowercasestring(x)
	    return x 

	def removenonsensewords(text):
		d = enchant.Dict("en_US")
		tokens = nltk.word_tokenize(text)
		stemmed = []
		for token in tokens:
			if d.check(token):
				stemmed.append(token)

		return ' '.join(stemmed)

	def listofbadwords():
	    from nltk.corpus import stopwords
	    stopwords = stopwords.words('english')
	    monthnames = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
	    randomrepetitive = ['edu','unl','mt']
	    
	    totlist = stopwords + monthnames + randomrepetitive
	    return totlist
	def removebadwords(x,totlist):
    
	    wordlist = x.split()
	    wordlist = [word for word in wordlist if word.lower() not in totlist]
	    x = ' '.join(wordlist)
	    return x

	def replacewithstem(text):
	    tokens = nltk.word_tokenize(text)
	    stemmer = nltk.stem.porter.PorterStemmer()
	    
	    stemmed = []
	    for token in tokens:
	        stemmed.append(stemmer.stem(token))
	        
	    return ' '.join(stemmed)

def main():
	inputc = InputClass()
	inputc.category = 'Android'
	inputc.author_comment_karma = 50000
	inputc.author_link_karma = 34000
	inputc.created_utc = 1000000000
	inputc.num_comments = 20
	inputc.thumbnail = 'link'
	inputc.type_var = 'nsfw'
	inputc.author_is_gold = True
	inputc.title = 'This post is about an awesome Android application that is going to take over the world'


	pred = Prediction()
	pred.transform_data(inputc)
	print(pred.finalPrediction)

main()