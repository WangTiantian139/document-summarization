"""
TF-IDF algorithm

input: a list which includes
       stemmed and stop-word-free sentence sub-lists
       obtained from original documents
output: a list whose index indicates the sentences in
        original document and whose value is a sub-list
        of the TF-IDF values, where each value combining
        a unique word appeared in the doc in which the
        sentence comes.

the target of this algorithm is to obtain a matrix
where rows as sentences while columns as words.

formula:
 | for a word in the word list, whose:
 | TF_IDF_value = TF * IDF;
 | (scope: one doc)
 | TF_value = count_of_the_word / count_of_all_words;
 | (scope: all docs)
 | IDF_value = log(count_of_the_docs / (count_of_the_docs_containing_word + 1);

"""


import math
import numpy as np
from document_process import DocProcess


DOC_PATH = 'doc/unprocessed_data/d30045t/'
REFERENCE_PATH = 'doc/reference/'
DOCUMENTS = ('NYT19981125.0417',
             'NYT19981125.0433',
             'NYT19981126.0192',
             'NYT19981127.0203',
             'NYT19981127.0240',
             'NYT19981127.0256',
             'NYT19981127.0264',
             'NYT19981127.0289',
             'NYT19981127.0293',
             'NYT19981129.0113',
             )
STOP_WORD_LIST = 'stop-word-list.csv'



'''
input stemmed sentences in just one document
and return a list of TF values of words 
appeared in the doc, addition, a dictionary
of count of word appeared in the document
NOTE that the sentences must in one document
'''


def tf(sentences: list):
    # split into words and count. the word would be included
    # in dictionary count_words only if the length of the word is
    # bigger than 1 as well as it is unique in words
    count_word = {}
    tf_word = {}
    sum_count_words = 0
    for count_w in count_word.keys():
        sum_count_words += count_word[count_w]
    for w in count_word.keys():
        tf_word[w] = count_word[w] / sum_count_words
    return tf_word, count_word


'''
the main method of the TF-IDF algorithm
'''


def tf_idf(data: DocProcess):
    # a sentence-word (row as sentence) matrix, whose element is TF-IDF value
    s_w_matrix = np.zeros((data.sen_size_total(), data.word_size_total()))

    # transverse every sentence
    # the sentence index in multi-doc scope
    general_sen_idx = 0
    for doc_idx in range(data.doc_size()):
        # the number of all of words in a certain doc
        wrd_sz_total = data.word_size(doc_idx)
        for sen_idx in range(data.sen_size(doc_idx)):
            # the word index related to word list in data
            wrd_idx = 0
            for word in data.word_list:
                # the number of how many times the word appears
                # in a certain doc
                count = data.count_in_doc(word, doc_idx)
                # many of 0 cases, skip for the matrix has been initialized with 0
                if count == 0:
                    continue
                elif count > 0:
                    tf = count / wrd_sz_total
                    idf = math.log(data.doc_size() / (data.count_doc(word) + 1), 10)
                    s_w_matrix[general_sen_idx, wrd_idx] = tf * idf
                elif count == -1:
                    print('error. no such word', word)
                else:
                    print('error. counter error.')
                wrd_idx += 1
            general_sen_idx += 1
    # regard all sentences in docs as a long one,
    # and compute its TF-IDF value
    long_sen_vector = []
    word_size_total = data.word_size_total()
    for word in data.word_list:
        tf = data.count_total(word) / word_size_total
        idf = math.log(1 / (1 + 1))
        long_sen_vector.append(tf * idf)

    return s_w_matrix, np.array(long_sen_vector)


'''
cosine similarity algorithm 
'''


def cos_similarity(sentence1: np.ndarray, sentence2: np.ndarray):
    sen1_pro_sen2 = np.transpose(sentence1) * sentence2
    amp_sen1, amp_sen2 = np.fabs(sentence1), np.fabs(sentence2)
    return np.cos(sen1_pro_sen2 / (amp_sen1 * amp_sen2))


def summarize(doc_path: str, doc_list: tuple):
    data = DocProcess(doc_path, doc_list)
    s_w_matrix, long_sen_vector = tf_idf(data)
    # a list of cos similarity, whose element is a tuple of
    # cosine value and original sentence's rank in sentence-word
    # matrix
    cos_sim_lst = []
    for r in range(np.size(s_w_matrix, 0)):
        cosine = cos_similarity(s_w_matrix[r], long_sen_vector)
        cos_sim_lst.append((cosine, r))
    cos_sim_lst.sort()
    check_rank = 0
    current_rank = 1
    summary = [data.abstract(cos_sim_lst[check_rank][1])]
    while current_rank != np.size(s_w_matrix, 0):
        if summary.__sizeof__() >= 665:
            break
        else:
            pass
        # compare 2 sentences
        cmp = cos_similarity(s_w_matrix[cos_sim_lst[current_rank][1]]
                             , s_w_matrix[cos_sim_lst[check_rank][1]])
        if cmp < 0.3:
            summary.append(data.abstract(cos_sim_lst[current_rank][1]))
            check_rank = current_rank
            current_rank = check_rank + 1
        else:
            current_rank += 1
    return summary


'''
//  main  //
'''

if __name__ == '__main__':
    summary = summarize(doc_path=DOC_PATH,
                        doc_list=DOCUMENTS)
    print(summary)
