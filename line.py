#!/usr/bin/env python3
import glob, os,re,math
from collections import OrderedDict
from numpy import dot
from numpy.linalg import norm
from stemming.porter2 import stem
from stop_words import get_stop_words
import pickle
from operator import itemgetter


stop_words = get_stop_words('en')
p = ['to','and','it','the']
for i in p:
    stop_words.append(i)
stop_words.remove('not')
stop_words.remove('be')
stop_words.remove('no')
print(stop_words)
