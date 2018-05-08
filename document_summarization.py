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
import convert_format_pyrouge as cfp

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


'''
the main method of the TF-IDF algorithm
'''


def tf_idf(data: DocProcess):
    # a sentence-word (row as sentence) matrix, whose element is TF-IDF value
    s_w_matrix = np.zeros((data.sen_size_total(), data.word_list.__len__()))

    # transverse every sentence
    # the sentence index in multi-doc scope
    general_sen_idx = 0
    for doc_idx in range(data.doc_size()):
        # the number of all of words in a certain doc
        for sen_idx in range(data.sen_size(doc_idx)):
            wrd_sz_sen = data.sen_word_size(doc_idx, sen_idx)
            for word in data.word_list:
                # the number of how many times the word appears
                # in a certain sentence
                count = data.count_in_sen(word, doc_idx, sen_idx)
                # many of 0 cases, skip for the matrix has been initialized with 0
                if count == 0:
                    continue
                elif count > 0:
                    tf = count / wrd_sz_sen
                    # log makes the idf value too small, so try to remove it
                    idf = math.log(data.sen_size(doc_idx) / (
                            data.count_sen_containing_word(word) + 1))
                    # the word index related to word list in data
                    wrd_idx = data.word_list.index(word)
                    s_w_matrix[general_sen_idx, wrd_idx] = tf * idf
                elif count == -1:
                    print('error. no such word ', word)
                else:
                    print('error. counter error.')
            general_sen_idx += 1
        print('doc ', doc_idx, ' completed.')
    # regard all sentences in docs as a long one,
    # and compute its TF-IDF value
    long_sen_vector = []
    word_size_total = data.word_size_total()
    for word in data.word_list:
        tf = data.count_total_in_doc(word) / word_size_total
        idf = math.log(1 / (1 + 1))
        long_sen_vector.append(tf * idf)

    return s_w_matrix, np.array(long_sen_vector)


'''
cosine similarity algorithm 
'''


def cos_similarity(sentence1: np.ndarray, sentence2: np.ndarray):
    sen1_pro_sen2 = sentence1.dot(sentence2)
    amp_sen1, amp_sen2 = np.linalg.norm(sentence1), np.linalg.norm(sentence2)
    return sen1_pro_sen2 / (amp_sen1 * amp_sen2 + 1e-9)


def write_for_test(m: np.ndarray, v: np.ndarray):
    # create file and write arrays separated by space to the file
    m.tofile('sentence-to-word-matrix.cache', ' ', '%.5f')
    v.tofile('long-sentence-vector.cache', ' ', '%.5f')


def read_for_test():
    infile = open('sentence-to-word-matrix.cache', 'r')
    matrix = infile.read()
    infile.close()
    infile = open('long-sen-vector.cache', 'r')
    vector = infile.read()
    infile.close()
    return np.array(matrix), np.array(vector)


def summarize(doc_path: str, doc_list: tuple):
    data = DocProcess(doc_path, doc_list)
    s_w_matrix, long_sen_vector = tf_idf(data)
    # save the sentence to word matrix and read it for saving time
    write_for_test(s_w_matrix, long_sen_vector)
    # s_w_matrix, long_sen_vector = read_for_test()
    # a list of cos similarity, whose element is a tuple of
    # cosine value and original sentence's rank in sentence-word
    # matrix
    sim_lst = []
    for r in range(s_w_matrix.shape[0]):
        cosine = cos_similarity(s_w_matrix[r], long_sen_vector)
        sim_lst.append((cosine, r))
    # sort by big2small order
    sim_lst.sort(reverse=True)

    # there are some 0-based row, which are bugs
    base_rank = 0
    while sim_lst[base_rank][0] == 1:
        base_rank += 1
    current_rank = base_rank + 1
    summary = [data.abstract(sim_lst[base_rank][1])]
    sum_size = summary[-1].__len__()
    while current_rank != sim_lst.__len__():
        if sum_size >= 665:
            break
        else:
            pass
        # compare 2 sentences
        sim = cos_similarity(s_w_matrix[sim_lst[current_rank][1]]
                             , s_w_matrix[sim_lst[base_rank][1]])
        # change the coefficient to change the summarization all by mind
        if sim < 0.7:
            summary.append(data.abstract(sim_lst[current_rank][1]))
            base_rank = current_rank
            current_rank = base_rank + 1
            sum_size += summary[-1].__len__()
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
    outfile = open('doc/systems/04systems/D30045.M.100.T.TT', 'w')
    for sentence in summary:
        outfile.write(sentence + '.\n')
    outfile.close()
    cfp.convert2rouge_format()
