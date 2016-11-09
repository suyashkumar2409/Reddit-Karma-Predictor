import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import string
import re
import nltk
import enchant
import sklearn
import * from Prediction

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