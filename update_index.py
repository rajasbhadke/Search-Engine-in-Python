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
for i in p:
    stop_words.append(i)
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

os.chdir("/home/rajas/PDF_spider/all_text")

with open('file_names.pickle', 'rb') as handle:
    file_names = pickle.load(handle)

with open('final_inverted_index.pickle', 'rb') as handle:
    inverted_index = pickle.load(handle)

with open('file_encoding.pickle', 'rb') as handle:
    file_encoding = pickle.load(handle)

with open('inverted_index_line.pickle', 'rb') as handle:
    inverted_index_line = pickle.load(handle)

def get_new_files():

    new_files = []
    os.chdir("/home/rajas/PDF_spider/all_text")
    for file in glob.glob("*.txt"):
        if file not in file_names and file not in new_files:
            new_files.append(file)
            file_names.append(file)

    return new_files

def get_deleted_files():

    deleted_files = []
    all_files = []
    os.chdir("/home/rajas/PDF_spider/all_text")
    for file in glob.glob("*.txt"):
        all_files.append(file)

    for i in file_names:
        if i not in all_files:
            deleted_files.append(i)
            file_names.remove(i)

    return deleted_files

def index_after_delete(deleted,inverted_index):

    for i in list(inverted_index.keys()):
        for j in deleted:
            if j in inverted_index[i].keys():
                temp = inverted_index[i].pop(j)
        if bool(inverted_index[i]) is False:
            temp = inverted_index.pop(i)

    return inverted_index
#this process creates a dictionary that maps pdfs to its words
def get_inverted_index(filenames,start):

    for idx,file in enumerate(filenames):
        try:
            idx = str(idx+start+1)
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

    last = int(list(file_encoding.keys())[-1])
    files_as_vectors = {str(key) : None for key in range(last+1)}
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
        # print(i)
        if files_as_vectors[i] is not None:
            for idx,value in enumerate(files_as_vectors[i]):
                doc_freq = len(inverted_index[words_to_num[files_as_vectors[i][idx][0]]].keys())
                idf = total_docs/doc_freq
                idf = math.log(idf)
                tf_idf = files_as_vectors[i][idx][1]*idf
                tf_idf = float('{:.3f}'.format(tf_idf))
                files_as_vectors[i][idx][1] = tf_idf

    return files_as_vectors


new_files = get_new_files()
deleted_files = get_deleted_files()
flag = 0

if deleted_files:
    flag = 1
    #update inverted index
    deleted_codes = []
    for i in file_encoding.keys():
        if file_encoding[i] in deleted_files:
            # print(i)
            deleted_codes.append(i)

    inverted_index = index_after_delete(deleted_codes,inverted_index)
    # print(inverted_index)
    #change encodings
    for i in list(file_encoding.keys()):
        if file_encoding[i] in deleted_files:
            # print(file_encoding[i])
            temp = file_encoding.pop(i)


if new_files:
    flag = 1
    #update encodings
    start = int(list(file_encoding.keys())[-1])
    for idx,value in enumerate(new_files):
        temp = str(idx+start+1)
        file_encoding[temp] = value

    #update inverted index
    inverted_index = get_inverted_index(new_files,start)

if flag == 1:
    files_as_vectors = get_files_as_vectors(file_names,inverted_index)

with open('final_inverted_index.pickle', 'wb') as handle:
    pickle.dump(inverted_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('files_as_vectors.pickle', 'wb') as handle:
    pickle.dump(files_as_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('file_names.pickle', 'wb') as handle:
    pickle.dump(file_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('file_encoding.pickle', 'wb') as handle:
    pickle.dump(file_encoding, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('inverted_index_line.pickle', 'wb') as handle:
    pickle.dump(inverted_index_line, handle, protocol=pickle.HIGHEST_PROTOCOL)
