#!/usr/bin/env python3
import glob, os,re,math
from collections import OrderedDict
from numpy import dot
from numpy.linalg import norm
from stemming.porter2 import stem
from stop_words import get_stop_words
from bisect import *
import pickle
from operator import itemgetter
# from autocorrect import spell

stop_words = get_stop_words('en')
p = ['to','and','it','the']
for i in p:
    stop_words.append(i)
stop_words.remove('not')
stop_words.remove('be')
stop_words.remove('no')


filenames = []
os.chdir("/home/rajas/PDF_spider/all_text")
with open('file_names.pickle', 'rb') as handle:
    filenames = pickle.load(handle)

with open('final_inverted_index.pickle', 'rb') as handle:
    inverted_index = pickle.load(handle)

with open('files_as_vectors.pickle', 'rb') as handle:
    files_as_vectors = pickle.load(handle)

with open('file_encoding.pickle', 'rb') as handle:
    file_encoding = pickle.load(handle)

with open('inverted_index_line.pickle', 'rb') as handle:
    inverted_index_line = pickle.load(handle)


def get_all_documents(wordList,inverted_index):
    lol = []
    sorted_list = []
    for i in wordList:
        if i in inverted_index.keys():
            temp_list = list(inverted_index[i].keys())
            temp_list = [int(x) for x in temp_list]
            sorted_list.append((len(temp_list),temp_list))
        else:
            return []

    sorted_list = sorted(sorted_list,key = lambda x:x[0])

    return [x[1] for x in sorted_list]


def merge_documents(lol,mode):


    if mode == 0:
        flag=0
        result = []
        j = 0
        if len(lol) == 0:
            return []
        if len(lol) == 1:
            return lol[0]
        while j < len(lol[0]):
            idx = 1
            while idx < len(lol) and bisect_left(lol[idx],lol[0][j]) < len(lol[idx]) and lol[idx][bisect_left(lol[idx],lol[0][j])] == lol[0][j]:
                idx = idx + 1
                if idx == len(lol):
                    flag = 1
                    result.append(lol[0][j])
                    break

            if idx < len(lol):
                if bisect_left(lol[idx],lol[0][j]) < len(lol[idx]):
                    j = bisect_left(lol[0],lol[idx][bisect_left(lol[idx],lol[0][j])])
                else:
                    j = j +1
            else:
                j = j + 1

        return result

    else:
        flag=0
        result = []
        j = 0
        if len(lol) == 0:
            return []
        if len(lol) == 1:
            return lol[0]
        while j < len(lol[0]):
            idx = 1
            while idx < len(lol) and bisect_left(lol[idx],lol[0][j]+idx) < len(lol[idx]) and lol[idx][bisect_left(lol[idx],lol[0][j]+idx)] == lol[0][j]+idx:
                idx = idx + 1
                if idx == len(lol):
                    flag = 1
                    result.append(lol[0][j])
                    break

            if idx < len(lol):
                if bisect_left(lol[idx],lol[0][j]+idx) < len(lol[idx]):
                    j = bisect_left(lol[0],lol[idx][bisect_left(lol[idx],lol[0][j]+idx)]-idx)
                    # print(j)
                else:
                    j = j +1
            else:
                j = j + 1

        return result

#process basic query - returns list of all documents having one of the word in query
def any_word_query(wordList,inverted_index):

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
            intersection_list = merge_documents(filenames_dict[i],1)
            # print(intersection_list)
            if len(intersection_list) != 0:
                basic_query.append(i)
    return basic_query


def query_as_vector(inverted_index,wordList,filenames):

    query_as_vector = []
    total_docs = len(filenames)
    wordlist_rms = {key:0 for key in wordList}
    for i in wordList:
        tf = 0
        if i in inverted_index.keys():

            for j in inverted_index[i].keys():
                tf = tf + len(inverted_index[i][j])
        wordlist_rms[i] = tf
    rms = 0
    for i in wordlist_rms.keys():
        rms = rms + (wordlist_rms[i]*wordlist_rms[i])
    rms = math.sqrt(rms)
    if rms == 0:
        return []
    for i in wordlist_rms.keys():
        wordlist_rms[i] = wordlist_rms[i]/rms

    count = 0
    for i in inverted_index.keys():
        if i in wordList:
            freq = len(inverted_index[i].keys())
            idf = total_docs/freq
            idf = math.log(idf)
            tf_idf = idf*wordlist_rms[i]
            query_as_vector.append([count,tf_idf])
        count = count + 1

    return query_as_vector

def calculate_dot(vector_a,vector_b):
    i = 0
    j = 0
    dot_product = 0
    while i < len(vector_a) and j < len(vector_b):
        if vector_a[i][0] == vector_b[j][0]:
            dot_product = dot_product + vector_a[i][1]*vector_b[j][1]
            i = i + 1
            j = j + 1

        elif vector_a[i][0] > vector_b[j][0]:
            j = j + 1

        else:
            i = i + 1

    return dot_product

def final_result_with_ranking(query_as_vector,files_as_vectors,phrase_query):

    final_result = {key : None for key in phrase_query}
    for i in phrase_query:
        vector_a = [query_as_vector[x][1] for x in range(len(query_as_vector))]
        vector_b = [files_as_vectors[i][x][1] for x in range(len(files_as_vectors[i]))]
        cos_sim = calculate_dot(query_as_vector, files_as_vectors[i])/(norm(vector_a)*norm(vector_b))
        final_result[i] = cos_sim
    final_result = OrderedDict(sorted(final_result.items(), key=itemgetter(1)))
    return list(final_result.keys())[::-1]

def get_line_numbers(result,wordList):

    result_with_lines = {file_encoding[key] : None for key in result}
    for i in result:
        for j in wordList:
            if i in inverted_index_line[j].keys():
                if result_with_lines[file_encoding[i]] is not None:
                    result_with_lines[file_encoding[i]].append(inverted_index_line[j][i])
                else:
                    result_with_lines[file_encoding[i]] = [inverted_index_line[j][i],]
            else:
                if result_with_lines[file_encoding[i]] is not None:
                    result_with_lines[file_encoding[i]].append([])
                else:
                    result_with_lines[file_encoding[i]] = [[],]

    return result_with_lines

print("1 : any of the words")
print("2 : all of the words")
print("3 : exact phrase")
while True:
    print("select the type of query:")
    select = input()
    if select=="-1":
        break
    print("enter the query:")
    query = input()
    query = query.lower()
    temp_wordList = re.sub("[^\w]", " ",  query).split()
    wordList = []
    for i in temp_wordList:
        if i not in stop_words:
            wordList.append(i)

    if len(wordList) == 0:
        print("Enter a valid query")

    else:
        if select == "1":
            any_query = any_word_query(wordList,inverted_index)
            query_as_v = query_as_vector(inverted_index,wordList,filenames)
            final_result = final_result_with_ranking(query_as_v,files_as_vectors,any_query)
            # print(final_result)
            # print(file_encoding)
            line_num =  get_line_numbers(final_result,wordList)
            if line_num:
                for i,j in line_num.items():
                    print(i)
                    print(j)
            else:
                print("No match found")
        elif select == "2":
             lol = get_all_documents(wordList,inverted_index)
             conjuction = merge_documents(lol,0)
             conjuction = [str(x) for x in conjuction]
             query_as_v = query_as_vector(inverted_index,wordList,filenames)
             final_result = final_result_with_ranking(query_as_v,files_as_vectors,conjuction)
             line_num =  get_line_numbers(final_result,wordList)
             if line_num:
                 for i,j in line_num.items():
                     print(i)
                     print(j)
             else:
                 print("No match found")
        elif select == "3":

            phrase_query = basic_phrase_query(wordList,inverted_index)
            # print(phrase_query)
            query_as_v = query_as_vector(inverted_index,wordList,filenames)
            final_result = final_result_with_ranking(query_as_v,files_as_vectors,phrase_query)
            # print(final_result)
            line_num =  get_line_numbers(final_result,wordList)
            for i in line_num.keys():
                line_num[i] = merge_documents(line_num[i],0)
            if line_num:
                for i,j in line_num.items():
                    print(i)
                    print(j)
            else:
                print("No match found")
