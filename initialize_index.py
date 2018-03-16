#!/usr/bin/env python3
import glob, os,re,math
from collections import OrderedDict
from numpy import dot
from numpy.linalg import norm
from stemming.porter2 import stem
from stop_words import get_stop_words
import pickle
from operator import itemgetter
import json

stop_words = get_stop_words('en')
p = ['to','and','it','the']
for i in p:
    stop_words.append(i)
stop_words.remove('not')
stop_words.remove('be')
stop_words.remove('no')


#this process creates a dictionary that maps pdfs to its words
inverted_index_line = {}
def get_inverted_index(filenames):
    inverted_index = {}
    for idx,file in enumerate(filenames):
        try:
            idx = str(idx)
            pattern = re.compile('[\W_]+')
            file_as_string = open(file, 'r')
            line_num = 0
            pos = 0
            for line in file_as_string.readlines():
                line_num += 1
                line = line.lower()
                line = pattern.sub(' ',line)
                re.sub(r'[\W_]+','', line)
                for word in line.split():
                    if word not in stop_words:
                        # word = stem(word)
                        if word in inverted_index_line.keys():
                            if idx in inverted_index_line[word].keys():
                                if inverted_index_line[word][idx][-1] != line_num:
                                    inverted_index_line[word][idx].append(line_num)
                            else:
                                inverted_index_line[word][idx] = [line_num,]
                        else:
                            inverted_index_line[word] = {idx : [line_num,]}

                        if word in inverted_index.keys():
                            if idx in inverted_index[word].keys():
                                inverted_index[word][idx].append(pos)
                            else:
                                inverted_index[word][idx] = [pos,]
                        else:
                            inverted_index[word] = {idx:[pos,]}
                        pos = pos + 1
        except UnicodeDecodeError:
            pass

    return inverted_index


def get_files_as_vectors(filenames,inverted_index):

    files_as_vectors = {str(key) : None for key in range(len(filenames))}
    words_to_num = {}
    count =  0
    total_docs = len(filenames)
    for i in inverted_index.keys():
        words_to_num[count] = i
        for key,value in inverted_index[i].items():
            if files_as_vectors[key] is not None:
                files_as_vectors[key].append([count,len(value)])
            else:
                files_as_vectors[key] = [[count,len(value)],]

        count = count + 1
    # print(files_as_vectors.keys())
    for i in files_as_vectors.keys():
        rms = 0
        if files_as_vectors[i] is not None:
            for j in files_as_vectors[i]:
                rms = rms + j[1]*j[1]
        rms = math.sqrt(rms)
        if rms !=0 :
            for idx in range(len(files_as_vectors[i])):
                files_as_vectors[i][idx][1] = files_as_vectors[i][idx][1]/rms
    # print(files_as_vectors)
    for i in files_as_vectors.keys():
        for idx,value in enumerate(files_as_vectors[i]):
            doc_freq = len(inverted_index[words_to_num[files_as_vectors[i][idx][0]]].keys())
            idf = total_docs/doc_freq
            idf = math.log(idf)
            tf_idf = files_as_vectors[i][idx][1]*idf
            tf_idf = float('{:.10f}'.format(tf_idf))
            # print(tf_idf)
            files_as_vectors[i][idx][1] = tf_idf

    return files_as_vectors


filenames = []

os.chdir("/home/rajas/PDF_spider/all_text")
for file in glob.glob("*.txt"):
    filenames.append(file)

#mapping the filenames to a unique number
file_encoding = {}
for idx,value in enumerate(filenames):
    idx=str(idx)
    file_encoding[idx] = value
inverted_index = get_inverted_index(filenames)
files_as_vectors = get_files_as_vectors(filenames,inverted_index)
# print(files_as_vectors)

with open('final_inverted_index.pickle', 'wb') as handle:
    pickle.dump(inverted_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('files_as_vectors.pickle', 'wb') as handle:
    pickle.dump(files_as_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('file_names.pickle', 'wb') as handle:
    pickle.dump(filenames, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('file_encoding.pickle', 'wb') as handle:
    pickle.dump(file_encoding, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('inverted_index_line.pickle', 'wb') as handle:
    pickle.dump(inverted_index_line, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('out.txt', 'w') as f:
#     print('Filename:', inverted_index, file=f)
# print(phrase_query)
# print(final_result)
