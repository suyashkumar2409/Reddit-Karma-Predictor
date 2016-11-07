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


	def __init__(self):
		class1 = pickleHandle.load_object('class1.pkl')
		class2 = pickleHandle.load_object('class2.pkl')


	def one_hot_conv(self, df, category,string):
		train_df = pd.read_pickle('IR-data.p')
		lis = train_df[string].unique()

		for col in lis:
			if col == category:
				df[string + "_" + col] = 1
			else:
				df[string + "_" + col] = 0


	def transform_data(self,test,input):
		one_hot_lis = ['Category','author_is_gold','thumbnail','type']

		this.one_hot_conv(test, input.category, 'Category')
		this.one_hot_conv(test, input.author_is_gold, 'author_is_gold')
		this.one_hot_conv(test, input.thumbnail, 'thumbnail')
		this.one_hot_conv(test, input.type, 'type')



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