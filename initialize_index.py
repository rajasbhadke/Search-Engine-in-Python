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

stop_words = get_stop_words('en')
p = ['to','and','it','the']
stop_words.append(p)
#this process creates a dictionary that maps pdfs to its words

def process_files(filenames):
	file_to_terms = {}
	for idx,file in enumerate(filenames):
		try:
			pattern = re.compile('[\W_]+')
			file_as_string = open(file, 'r').read().lower()
			file_as_string = pattern.sub(' ',file_as_string)
			re.sub(r'[\W_]+','', file_as_string)
			for word in file_as_string.split():
				if word not in stop_words:
					if idx in file_to_terms.keys():
						file_to_terms[idx].append(stem(word))
					else:
						file_to_terms[idx] = [stem(word),]
		except UnicodeDecodeError:
			pass

	return file_to_terms

#this process maps words from single file to their positions

def map_words_to_pos(word_list):
    words_to_pos = {}
    for index,value in enumerate(word_list):

        if value in words_to_pos.keys():
            words_to_pos[value].append(index)
        else:
            words_to_pos[value] = [index]

    return words_to_pos

#this process returns the final inverted-index
def get_inverted_index(temp_index):

    inverted_index = {}
    for key,value in temp_index.items():
        for i,j in value.items():
            if i in inverted_index.keys():
                if key in inverted_index[i].keys():
                    inverted_index[i][key].append(j)
                else:
                    inverted_index[i][key] = j
            else:
                inverted_index[i] = {key:j}

    return inverted_index

#calculate the term-frequency and return every document as a vector of unique words
def term_frequency(filenames,inverted_index):

	files_as_vectors = {key : [0]*len(inverted_index.keys()) for key in range(len(filenames))}
	count =  0
	for i in inverted_index.keys():
		for key,value in inverted_index[i].items():
			files_as_vectors[key][count] = len(value)
		count = count + 1

	for i in files_as_vectors.keys():
		rms = 0
		for j in files_as_vectors[i]:
			rms = rms + j*j
		rms = math.sqrt(rms)
		if rms !=0 :
			for idx in range(len(files_as_vectors[i])):
				files_as_vectors[i][idx] = files_as_vectors[i][idx]/rms

	return files_as_vectors

def inverse_document_frequency(files_as_vectors,inverted_index,filenames):

	count = 0
	for i in inverted_index.keys():

		doc_freq = len(inverted_index[i].keys())
		total_docs = len(filenames)
		idf = total_docs/doc_freq
		idf = math.log(idf)
		for j in files_as_vectors.keys():
			files_as_vectors[j][count] = files_as_vectors[j][count]*idf
			files_as_vectors[j][count] = float('{:.10f}'.format(files_as_vectors[j][count]))
		count = count + 1
	return files_as_vectors


filenames = []

os.chdir("/home/rajas/PDF_spider/all_text")
for file in glob.glob("*.txt"):
    filenames.append(file)

file_encoding = {}
for idx,value in enumerate(filenames):
	file_encoding[idx] = value

file_to_terms = process_files(filenames)

temp_index = {}
for key,value in file_to_terms.items():

    words_to_pos = map_words_to_pos(value)
    temp_index[key] = words_to_pos

# print(temp_index)
inverted_index = get_inverted_index(temp_index)

files_as_vectors = term_frequency(filenames,inverted_index)
files_as_vectors = inverse_document_frequency(files_as_vectors,inverted_index,filenames)

with open('final_inverted_index.json', 'w') as fp:
    json.dump(inverted_index, fp)

with open('files_as_vectors.json', 'w') as fp:
    json.dump(files_as_vectors, fp)

with open('file_names.json', 'w') as fp:
    json.dump(filenames, fp)

with open('file_encoding.json', 'w') as fp:
    json.dump(file_encoding, fp)

# print(inverted_index)
# print(phrase_query)
# print(final_result)
