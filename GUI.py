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
from Prediction import *

import Tkinter as tk


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

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        FRAME = tk.Frame.__init__(self, parent, *args, **kwargs)

        self.entries = {}

        tk.Label(FRAME, text="TITLE").pack()
        title= self.entries['title']= tk.Entry(FRAME)
        title.pack()
        
        tk.Label(FRAME, text='SUBREDDIT').pack()
        _s = self.entries['category'] = tk.StringVar()
        _s.set('trippy')
        OPTIONS = ['trippy', 'AskReddit', 'bestof', 'rage', 'creepy', 'WTF', 'movies', 'Music', 
        'funny', 'facepalm', 'Jokes', 'comics', 'pcs', 'videos', 'askscience', 'science', 
        'space', 'trees', 'cats', 'food', 'guns', 'teenagers', 'sex', 'drunk', 'Art', 
        'lifehacks', 'DIY', 'news', 'Conservative', 'news', 'politics', 'worldnews', 'nba', 
        'soccer', 'MMA', 'technology', 'Android', 'Bitcoin', 'programming', 'apple']

        option =tk.OptionMenu(FRAME, _s,*OPTIONS)
        option.pack()


        tk.Label(FRAME, text='THUMBNAIL').pack()
        _s = self.entries['thumbnail'] = tk.StringVar()
        _s.set('default')
        OPTIONS = ['default','nsfw','link','self']

        option =tk.OptionMenu(FRAME, _s,*OPTIONS)
        option.pack()

        tk.Label(FRAME, text='TYPE').pack()
        _i= self.entries['type_var'] = tk.StringVar(value='img')
        _ = tk.Radiobutton(FRAME, text='IMG',value='img', variable=_i)
        _.pack()
        _ = tk.Radiobutton(FRAME, text='VID', variable=_i, value='vid')
        _.pack()
        _ = tk.Radiobutton(FRAME, text='TXT', variable=_i, value='txt')
        _.pack()

        
        tk.Label(FRAME, text='COMMENT KARMA').pack()
        _i = self.entries['author_comment_karma'] = tk.IntVar()
        cmt_ = tk.Entry(FRAME, textvariable = _i)
        cmt_.pack()
        tk.Label(FRAME, text='LINK KARMA').pack()
        _i = self.entries['author_link_karma'] = tk.IntVar()
        link_ = tk.Entry(FRAME, textvariable = _i)
        link_.pack()
        

        tk.Label(FRAME, text="IS_GOLD").pack()
        
        _i= self.entries['author_is_gold'] = tk.BooleanVar()
        _ = tk.Radiobutton(FRAME, text='Yes',value=True, variable=_i)
        _.pack()
        _ = tk.Radiobutton(FRAME, text='No', variable=_i, value=False)
        _.pack()


        tk.Label(FRAME, text="CREATED").pack()
        created = self.entries['created_utc']= tk.Entry(FRAME)
        created.pack()

        tk.Label(FRAME, text="NUMBER OF COMMENTS").pack()
        _i = self.entries['num_comments'] = tk.IntVar()
        num_comments = tk.Entry(FRAME, textvariable=_i)
        num_comments.pack()
       

        tk.Button(FRAME, text='Predict Popularity', command=self.predict, width=25).pack()
        self.score = tk.StringVar(value='SCORE: ')
        _= tk.Label(FRAME, text='SCORE: ', textvariable=self.score)
        _.pack(anchor='c', pady=30)


    def set_score(self, value):
        self.score.set('SCORE: %d'%value)

    def predict(self):
            #instal tkinter
            s= Input()
            for x in self.entries:
                setattr(s,x,self.entries[x].get())
            #call your function with s object here or use your class instead
            s.created_utc = int(s.created_utc)

            pred = Prediction()
            pred.transform_data(s)
            finalAns = (int(round(pred.finalPrediction()[0])))
            self.set_score(finalAns)

    def say_hello(self):
    	s= Input()
    	for x in self.entries:
			setattr(s,x,self.entries[x].get())

        s.author_comment_karma = float(s.author_comment_karma)
        s.author_link_karma = float(s.author_link_karma)
        s.num_comments = int(s.num_comments)
        
        attrs = vars(s)
        print ', '.join("%s: %s" % item for item in attrs.items())

        #print(s)


		#call your function with s object here or use your class instead
		#also instal tkinter




if __name__ == '__main__':
    root = tk.Tk()
    root.config(cursor='hand1')
    MainApplication(root, height=400, width=1000,)
    root.title('Reddit Application')
    root.resizable(width=False, height=False)
root.mainloop()