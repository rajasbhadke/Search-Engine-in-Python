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

filenames = []
os.chdir("/home/rajas/PDF_spider/all_text")
for file in glob.glob("*.txt"):
    filenames.append(file)


with open('final_inverted_index.json') as json_data:
    inverted_index = json.load(json_data)

with open('files_as_vectors.json') as json_data:
    files_as_vectors = json.load(json_data)

with open('file_encoding.json') as json_data:
    file_encoding = json.load(json_data)

#process basic query - returns list of all documents having one of the word in query
def basic_word_query(wordList,inverted_index):

    basic_query = []
    for i in wordList:
        if i in inverted_index.keys():
            for key in inverted_index[i].keys():
                basic_query.append(key)
    return basic_query

#process basic query - returns list of all documents having entire phrase in query
def basic_phrase_query(wordList,inverted_index):

	#improve the list intersection function
	basic_query = []
	filenames_dict = {}
	for i in wordList:
		if i in inverted_index.keys():
			for j in inverted_index[i].keys():
				if j in filenames_dict.keys():
					filenames_dict[j].append(inverted_index[i][j])
				else:
					filenames_dict[j] = [inverted_index[i][j],]

	# print(filenames_dict)

	for i in filenames_dict.keys():
		if len(filenames_dict[i]) == len(wordList):
			for idx,val in enumerate(filenames_dict[i]):
				val[:] = [x - idx for x in val]
			intersection_list = list(reduce(set.intersection, [set(item) for item in filenames_dict[i] ]))
			if len(intersection_list) != 0:
				basic_query.append(i)
	return basic_query




def query_as_vector(inverted_index,wordList,filenames):

	temp_dict = {}
	temp_dict2 = {}
	idf_dict = {key : 0 for key in range(len(inverted_index.keys()))}
	count = 0
	total = len(filenames)
	for i in inverted_index.keys():
		temp_dict[i] = count
		count = count + 1
		num = 0
		for j in inverted_index[i].keys():
			num = num + len(inverted_index[i][j])
		temp_dict2[i] = num

	query_as_vector = [0]*len(inverted_index.keys())
	for i in wordList:
		if i in inverted_index.keys():
			query_as_vector[temp_dict[i]] = temp_dict2[i]
			doc_freq = len(inverted_index[i].keys())
			idf = total/doc_freq
			idf_dict[temp_dict[i]] = math.log(idf)
	rms = 0
	for j in query_as_vector:
		rms = rms + j*j
	rms = math.sqrt(rms)
	if rms !=0 :
		for idx in range(len(query_as_vector)):
			query_as_vector[idx] = query_as_vector[idx]/rms
			query_as_vector[idx] = query_as_vector[idx]*idf_dict[idx]
			query_as_vector[idx] = float('{:.10f}'.format(query_as_vector[idx]))

	return query_as_vector

def final_result_with_ranking(query_as_vector,files_as_vectors,phrase_query):

	final_result = {key : None for key in phrase_query}
	for i in phrase_query:
		cos_sim = dot(query_as_vector, files_as_vectors[i])/(norm(query_as_vector)*norm(files_as_vectors[i]))
		final_result[i] = cos_sim
	final_result = OrderedDict(sorted(final_result.items(), key=itemgetter(1)))
	return [file_encoding[x] for x in list(final_result.keys())]


query = input()
query = query.lower()
temp_wordList = re.sub("[^\w]", " ",  query).split()
wordList = []
for i in temp_wordList:
    if i not in stop_words:
        wordList.append(stem(i))
phrase_query = basic_phrase_query(wordList,inverted_index)
query_as_v = query_as_vector(inverted_index,wordList,filenames)
final_result = final_result_with_ranking(query_as_v,files_as_vectors,phrase_query)
print(final_result)
