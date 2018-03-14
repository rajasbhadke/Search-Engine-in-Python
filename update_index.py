#!/usr/bin/env python3
import time
import glob, os,re,math
from collections import OrderedDict
from numpy import dot
from numpy.linalg import norm
from stemming.porter2 import stem
from stop_words import get_stop_words
from operator import itemgetter
import pickle

stop_words = get_stop_words('en')
p = ['to','and','it','the']
stop_words.append(p)
stop_words.remove('not')
stop_words.remove('be')
stop_words.remove('no')
# def get_modified_files():
#     _dir = os.getcwd()
#     new_files = []
#     new_files = list((fle for rt, _, f in os.walk(_dir) for fle in f if time.time() - os.stat(
#         os.path.join(rt, fle)).st_mtime < 30))
#
#
#     with open('file_names.json') as json_data:
#         file_names = json.load(json_data)
#
#     os.chdir("/home/rajas/PDF_spider/all_text")
#     for file in glob.glob("*.txt"):
#         if file not in file_names and file not in new_files:
#             new_files.append(file)
#
#     recent_files = []
#     for i in new_files:
#         if i[-3:] == "txt":
#             recent_files.append(i)
#
#     return recent_files

os.chdir("/home/rajas/Documents/PDF_spider_2/all_text")

with open('file_names.pickle', 'rb') as handle:
    file_names = pickle.load(handle)

with open('final_inverted_index.pickle', 'rb') as handle:
    inverted_index = pickle.load(handle)

with open('files_as_vectors.pickle', 'rb') as handle:
    files_as_vectors = pickle.load(handle)

with open('file_encoding.pickle', 'rb') as handle:
    file_encoding = pickle.load(handle)

with open('inverted_index_line.pickle', 'rb') as handle:
    inverted_index_line = pickle.load(handle)

def get_new_files():

    new_files = []
    os.chdir("/home/rajas/Documents/PDF_spider_2/all_text")
    for file in glob.glob("*.txt"):
        if file not in file_names and file not in new_files:
            new_files.append(file)

    return new_files

def get_deleted_files():

    deleted_files = []
    all_files = []
    os.chdir("/home/rajas/Documents/PDF_spider_2/all_text")
    for file in glob.glob("*.txt"):
        all_files.append(file)

    for i in file_names:
        if i not in all_files:
            deleted_files.append(i)

    return deleted_files


#this process creates a dictionary that maps pdfs to its words
def process_files(filenames,start):
    file_to_terms = {}
    for idx,file in enumerate(filenames):
        temp = str(idx+start+1)
        try:
            idx = str(idx)
            pattern = re.compile('[\W_]+')
            file_as_string = open(file, 'r')
            line_num = 0;
            for line in file_as_string.readlines():
                line_num += 1
                line = line.lower()
                line = pattern.sub(' ',line)
                re.sub(r'[\W_]+','', line)
                for word in line.split():
                    if word not in stop_words:
                        word = stem(word)
                        if word in inverted_index_line.keys():
                            if temp in inverted_index_line[word].keys():
                                if inverted_index_line[word][temp][-1] != line_num:
                                    inverted_index_line[word][temp].append(line_num)
                            else:
                                inverted_index_line[word][temp] = [line_num,]
                        else:
                            inverted_index_line[word] = {temp : [line_num,]}

                        if temp in file_to_terms.keys():
                            file_to_terms[temp].append(word)
                        else:
                            file_to_terms[temp] = [word,]
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
def term_frequency(filenames,inverted_index,last):

    files_as_vectors = {str(key) : None for key in range(last+1)}
    words_to_num = {}
    count =  0
    total_docs = len(filenames)
    for i in inverted_index.keys():
        words_to_num[count] = i
        for key,value in inverted_index[i].items():
            if files_as_vectors[key] is not None:
                files_as_vectors[key].append((count,len(value)))
            else:
                files_as_vectors[key] = [(count,len(value)),]

        count = count + 1

    for i in files_as_vectors.keys():
        rms = 0
        if files_as_vectors[i] is not None:
            for j in files_as_vectors[i]:
                rms = rms + j[1]*j[1]
        rms = math.sqrt(rms)
        if rms !=0 :
            for idx in range(len(files_as_vectors[i])):
                files_as_vectors[i][idx] = files_as_vectors[i][idx][1]/rms


    for i in files_as_vectors.keys():
        for idx,value in enumerate(files_as_vectors[i]):
            doc_freq = len(inverted_index[words_to_num[files_as_vectors[i][idx][0]]].keys())
            idf = total_docs/doc_freq
            idf = math.log(idf)
            tf_idf = files_as_vectors[i][idx][1]*idf
            tf_idf = float('{:.5f}'.format(tf_idf))
            files_as_vectors[i][idx][1] = tf_idf

    return files_as_vectors


new_files = get_new_files()
deleted_files = get_deleted_files()
# print(new_files)
if deleted_files:

    inverted_index = modify_index_for_deleted_files(inverted_index,deleted_files)


if new_files:

    new_file_encoding = {}
    start = int(list(file_encoding.keys())[-1])

    for idx,value in enumerate(new_files):
        temp = str(idx+start+1)
        new_file_encoding[temp] = value

    file_to_terms = process_files(new_files,start)
    temp_index = {}
    for key,value in file_to_terms.items():

        words_to_pos = map_words_to_pos(value)
        temp_index[key] = words_to_pos

    inverted_index = get_inverted_index(temp_index,inverted_index)
    # print(inverted_index)
    filenames = new_files + file_names
    last = int(list(new_file_encoding.keys())[-1])
    files_as_vectors = term_frequency(filenames,inverted_index,last)
    #print(files_as_vectors)

    #merged dictionary of new and old file encodings
    merged = {**file_encoding,**new_file_encoding}
    with open('final_inverted_index.pickle', 'wb') as handle:
        pickle.dump(inverted_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('files_as_vectors.pickle', 'wb') as handle:
        pickle.dump(files_as_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('file_names.pickle', 'wb') as handle:
        pickle.dump(filenames, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('file_encoding.pickle', 'wb') as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('inverted_index_line.pickle', 'wb') as handle:
        pickle.dump(inverted_index_line, handle, protocol=pickle.HIGHEST_PROTOCOL)
