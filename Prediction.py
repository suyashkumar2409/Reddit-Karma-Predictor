import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import string
import re
import nltk
import enchant
import sklearn

%matplotlib inline

class Input:
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

	def __init__(self):
		class1 = pickleHandle.load_object('class1.pkl')
		class2 = pickleHandle.load_object('class2.pkl')
		train_df = pd.read_pickle('IR-data.p')
		final_train_df = pd.read_pickle('PostClass2.p')

	def one_hot_conv(self, df, category,string):
		lis = train_df[string].unique()

		for col in lis:
			if col == category:
				df[string + "_" + col] = 1
			else:
				df[string + "_" + col] = 0

	def numerical_conv(self, df, input):
		attr = ['created_utc','title','ups','author_link_karma','author_comment_karma']

		avg = train_df[created_utc].mean()
        diff = train_df[created_utc].max() - train_df[created_utc].min()
        df[created_utc] = (input.created_utc - avg)/diff - final_train_df[created_utc].min()

		avg = train_df[title].mean()
        diff = train_df[title].max() - train_df[title].min()
        df[title] = (input.title - avg)/diff - final_train_df[title].min()
        
        avg = train_df[author_link_karma].mean()
        diff = train_df[author_link_karma].max() - train_df[author_link_karma].min()
        df[author_link_karma] = (input.author_link_karma - avg)/diff - final_train_df[author_link_karma].min()
        
        avg = train_df[author_comment_karma].mean()
        diff = train_df[author_comment_karma].max() - train_df[author_comment_karma].min()
        df[author_comment_karma] = (input.author_comment_karma - avg)/diff - final_train_df[author_comment_karma].min()

    def string_conv(self, df, input):
    	df['title_clean'] = textHandle.cleanstring(input.title)
    	df['title_nonsense'] = textHandle.removenonsensewords(df['title_clean'])
    	df['title_badwords'] = textHandle.removebadwords(df['title_nonsense'], textHandle.listofbadwords())

    	df['title_stemmed'] = textHandle.replacewithstem(df['title_badwords'])



	def transform_data(self,test,input):
		one_hot_lis = ['Category','author_is_gold','thumbnail','type']

		this.one_hot_conv(test, input.category, 'Category')
		this.one_hot_conv(test, input.author_is_gold, 'author_is_gold')
		this.one_hot_conv(test, input.thumbnail, 'thumbnail')
		this.one_hot_conv(test, input.type, 'type')

		this.numerical_conv(test,input)


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

	        #print(i)

	        #i=i+1

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