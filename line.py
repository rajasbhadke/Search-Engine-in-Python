#!/usr/bin/env python3
import glob, os,re,math
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from operator import itemgetter
from numpy import dot
from numpy.linalg import norm
import json
from stemming.porter2 import stem
from stop_words import get_stop_words

os.chdir("/home/rajas/PDF_spider/all_text")
with open('final_inverted_index.json') as json_data:
    inverted_index_line = json.load(json_data)

for i in inverted_index_line.keys():
    for j in inverted_index_line[i].keys():
        print(type(j))
        break
