import time
import glob, os,re,math
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from operator import itemgetter
from numpy import dot
from numpy.linalg import norm
import json

def get_modified_files():
    _dir = os.getcwd()
    new_files = []
    new_files = list((fle for rt, _, f in os.walk(_dir) for fle in f if time.time() - os.stat(
        os.path.join(rt, fle)).st_mtime < 30))


    with open('file_names.json') as json_data:
        file_names = json.load(json_data)

    os.chdir("/home/rajas/PDF_spider/all_text")
    for file in glob.glob("*.txt"):
        if file not in file_names and file not in new_files:
            new_files.append(file)

    recent_files = []
    for i in new_files:
        if i[-3:] == "txt":
            recent_files.append(i)

    return recent_files

#this process creates a dictionary that maps pdfs to its words
def process_files(filenames):
	file_to_terms = {}
	for file in filenames:
		pattern = re.compile('[\W_]+')
		file_to_terms[file] = open(file, 'r').read().lower()
		file_to_terms[file] = pattern.sub(' ',file_to_terms[file])
		re.sub(r'[\W_]+','', file_to_terms[file])
		file_to_terms[file] = file_to_terms[file].split()
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
def get_inverted_index(temp_index,inverted_index):

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

	files_as_vectors = {key : [0]*len(inverted_index.keys()) for key in filenames}
	count =  0
	for i in inverted_index.keys():
		for key,value in inverted_index[i].items():
			if key in filenames:
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
			if j in filenames:
				files_as_vectors[j][count] = files_as_vectors[j][count]*idf
				files_as_vectors[j][count] = float('{:.10f}'.format(files_as_vectors[j][count]))
		count = count + 1
	return files_as_vectors


new_files = get_modified_files()
if new_files:
    with open('final_inverted_index.json') as json_data:
        inverted_index = json.load(json_data)

    with open('files_as_vectors.json') as json_data:
        files_as_vectors = json.load(json_data)

    with open('file_names.json') as json_data:
        file_names = json.load(json_data)

    file_to_terms = process_files(new_files)
    temp_index = {}

    for key,value in file_to_terms.items():

        words_to_pos = map_words_to_pos(value)
        temp_index[key] = words_to_pos


    inverted_index = get_inverted_index(temp_index,inverted_index)
    # print(inverted_index)
    new_files_as_vectors = term_frequency(new_files,inverted_index)
    new_files_as_vectors = inverse_document_frequency(new_files_as_vectors,inverted_index,new_files)

    merged = {**files_as_vectors, **new_files_as_vectors}

    with open('final_inverted_index.json', 'w') as fp:
        json.dump(inverted_index, fp)

    with open('files_as_vectors.json', 'w') as fp:
        json.dump(merged, fp)

    filenames = new_files + file_names
    with open('file_names.json', 'w') as fp:
        json.dump(filenames, fp)
