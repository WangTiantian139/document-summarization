"""
        Document Process

This file is to define a class where is
storing the processed data from original documents.

The necessary components are
 (a) Original sentences which classified by documents,
 (b) Stop-word-free and stemmed sentences which signed
  with index ordered by where the original sentences is,
 (c) A dictionary where the unique words appeared in all
  of the documents which are regarded as keys, which
  indicated by the key is a numpy-array-based list.
  For one of that lists, the indexes relate to document's
  ones as well as elements are arrays of the counts of
  word ordered by sentence indexes.

However, it is easy for using to define the components
mentioned above as public ones.

Plus, some methods for using the private data are in need,
such as abstract the interested text from a document list,
which can defined as a protected method.

data structure

 doc : n(doc) x m(sen) (sen = {str})
   doc0[ [sen1, sen2, sen3, ... ,senk],
   doc1| [sen(k+1), sen(k+2), ... ]
   ... | [ ... ]
   docn] [ ... , senm]

 word_counter : q(wrd)
       { 'wrd0' : [ value_list0 ]
       | 'wrd1' : [ value_list1 ]
       |   ...  :      ...
       } 'wrdq' : [ value_listq ]

 value_list - for word_counter : n(doc) x m(sen) (sen = {int})
   doc0[ [sen1, sen2, sen3, ... ,senk],
   doc1| [sen(k+1), sen(k+2), ... ]
   ... | [ ... ]
   docn] [ ... , senm]

"""
import numpy as np

from poter_stemming import PorterStemmer

REFERENCE_PATH = ''
STOP_WORD_LIST = 'stop-word-list.csv'

'''
Split text in the document into sentences.return a list, which contains sub-lists 
of sentences in a document, of all the text of the documents in the DOCUMENTS tuple.  
NOTE that there are some sentences are not to be recognized such as ones end with '?' 
or end with '.\'\'', as well as some extra punctuation such as `` or '' are remained
too.  
'''


def split_sentences(path: str, documents: tuple):
    doc = []
    # transverse every document
    for f in documents:
        # open the current document
        infile = open(path + f, 'r')
        str_text = ''
        # skip line until the text start signal has been read
        while infile.readline() != '<TEXT>\n':
            continue
        # store all text sentences without '\n' into sentences
        while True:
            line = infile.readline()
            if line != '</TEXT>\n':
                str_text += line[:-1]  # the last char is always '\n'
            else:
                break
        sentences = str_text.split('. ')
        # if the last sentence is empty string, abandon it
        if sentences[-1] == '':
            sentences = sentences[:-1]
        doc.append(sentences)
        infile.close()
    return doc


'''
delete stop word in sentence as well as transform the sentence into 
lower case 
'''


def delete_stop_words(ori_sen: str):
    # for identifying the real stop word
    # from sub-string in a word which is as same
    # as some stop word, it is more simple to add
    # space in the front and at the end of the sentence,
    # while using ' ' + word + ' ' to identify the stop
    # word
    sentence = ' ' + ori_sen.lower() + ' '
    # read stop word list in file
    infile = open(REFERENCE_PATH + STOP_WORD_LIST)
    # read and transform into lower case
    lines = infile.read().lower()
    stop_word_list = lines.split(', ')
    # splitting into words is to avoid form deleting some
    # characters in a certain word which has a sub-string
    # containing the stop word
    words = sentence.split(' ')
    for w in words:
        if w in stop_word_list:
            sentence = sentence.replace(' ' + w + ' ', ' ')
        else:
            continue
    # remove the spaces in the beginning and at the end,
    # which were added in the beginning of this method
    return sentence[1:-1]


'''
use class PorterStemmer written by Porter
return a list of stemmed sentences 
'''


def porter_stemming(sentences: list):
    p = PorterStemmer()
    new_sentences = []
    for s in sentences:
        output = ''
        word = ''
        for c in s:
            if c.isalpha():
                word += c.lower()
            else:
                if word:
                    output += p.stem(word, 0, len(word) - 1)
                    word = ''
                output += c.lower()
        new_sentences.append(output)
    return new_sentences


def count_word(sentence: str):
    w = ''
    counter_in_sen = {}
    for c in sentence:
        if c.isalpha():
            w += c.lower()
        else:
            if w.__len__() <= 1:
                continue
            if w not in counter_in_sen.keys():
                counter_in_sen[w] = 1
                w = ''
            else:
                counter_in_sen[w] += 1
                w = ''
    return counter_in_sen


class WordCounter:
    def __init__(self):
        self.cntr = {}

    def generate_value_list(self, doc: list, word: str):
        ori_cnt_lst = []
        for d in doc:
            counter_in_sen = [0 for i in range(d.__len__())]
            ori_cnt_lst.append(counter_in_sen)
        self.cntr[word] = ori_cnt_lst


class DocProcess:

    def __init__(self, doc_path: str, doc_list: tuple):
        # a list to store all sentences in every document
        self.ori_doc = []

        # a list of Stop-word-free and stemmed sentences which signed
        # with index ordered by where the original sentences is,
        self.processed_doc = []

        # a list of unique words
        self.word_list = []

        # a dictionary of the count of appeared words
        self.word_counter = WordCounter()

        # split text in doc into sentences
        self.ori_doc = split_sentences(path=doc_path, documents=doc_list)
        # delete stop words and stem remained words
        # store the processed data in processed_doc
        for old_doc in self.ori_doc:
            new_doc = []
            for s in old_doc:
                new_doc.append(delete_stop_words(s))
            new_doc = porter_stemming(new_doc)
            self.processed_doc.append(new_doc)

        # count words
        doc_idx = 0
        for pro_doc in self.processed_doc:
            sen_idx = 0
            for sen in pro_doc:
                counter_in_sen = count_word(sen)
                for w in counter_in_sen.keys():
                    if w in self.word_counter.cntr.keys():
                        self.word_counter.cntr[w][doc_idx][sen_idx] += 1
                    else:
                        # if not add the word, init the value 2-level list with 0
                        self.word_counter.generate_value_list(self.processed_doc, w)
                        self.word_counter.cntr[w][doc_idx][sen_idx] += 1
                sen_idx += 1
            doc_idx += 1
        self.word_list = list(self.word_counter.cntr.keys())

    # count how many times the word appears in the sentence
    def count_in_sen(self, word, doc_index, sen_index):
        if word in self.word_counter.cntr.keys():
            return self.word_counter.cntr[word][doc_index][sen_index]
        else:
            return -1

    # count how many times the word appears in the doc[doc_index]
    def count_in_doc(self, word, doc_index):
        if word in self.word_counter.cntr.keys():
            sum_count = sum(list(self.word_counter.cntr[word][doc_index]))
            return sum_count
        else:
            return -1

    # count how many times the word appears in all docs
    def count_total_in_doc(self, word):
        if word in self.word_counter.cntr.keys():
            s = 0
            dual_l = self.word_counter.cntr[word]
            for l in dual_l:
                s += sum(l)
            return s
        else:
            return -1

    # count how many docs contain the word
    def count_doc_containing_word(self, word):
        if word in self.word_counter.cntr.keys():
            count = 0
            for doc_idx in range(self.doc_size()):
                # if the word was contained by the doc, there must be some sentence
                # contain the word. therefore the sum should greater than 0
                if sum(list(self.word_counter.cntr[word][doc_idx])):
                    count += 1
                else:
                    continue
            return count
        else:
            return -1

    def doc_size(self):
        return self.processed_doc.__len__()

    def sen_size(self, doc_index):
        return self.processed_doc[doc_index].__len__()

    def sen_size_total(self):
        sum = 0
        for doc_idx in range(self.doc_size()):
            sum += self.sen_size(doc_idx)
        return sum

    # count how many word there is in the processed doc,
    # in this case, we do not care if some word repeats
    def word_size(self, doc_index):
        s = 0
        for word in self.word_list:
            l = self.word_counter.cntr[word][doc_index]
            s += sum(list(l))
        return s

    # count how many word there is in all of the processed docs
    def word_size_total(self):
        s = 0
        for word in self.word_list:
            s += self.count_total_in_doc(word)
        return s

    def abstract(self, sen_rank):
        for doc in self.ori_doc:
            if sen_rank >= doc.__len__():
                sen_rank -= doc.__len__()
            else:
                return doc[sen_rank]


def basic_test():
    SPLIT_CACHE = 'doc/cache/split.cache'
    DEL_STOP_WORD_CACHE = 'doc/cache/del_stop_word.cache'
    STEM_CACHE = 'doc/cache/stem.cache'
    # test split sentences
    outfile = open(SPLIT_CACHE, 'w')
    doc_index = 0
    # split the text in docs into sentences
    doc_sentences = split_sentences(DOC_PATH, DOCUMENTS)
    for sentences in doc_sentences:
        # output <index> to indicate the beginning of the current document
        outfile.write('<' + str(doc_index) + '>\n')
        for s in sentences:
            outfile.write(s + '\n')
        # output <\index> to indicate the end of the current document
        outfile.write('</' + str(doc_index) + '>\n')
        outfile.write('\n')
        doc_index += 1
    print('The unprocessed data has been split into sentences by space. ')
    print('The result is output in ' + SPLIT_CACHE)
    outfile.close()
    # test deleting stop word
    infile = open(SPLIT_CACHE, 'r')
    outfile = open(DEL_STOP_WORD_CACHE, 'w')
    lines = infile.read()
    lines = lines.split('\n')
    # delete stop words
    for l in lines:
        sentence = delete_stop_words(l)
        outfile.write(sentence + '\n')
    print('The split sentences has been free of stop words. ')
    print('The result is output in ' + DEL_STOP_WORD_CACHE)
    outfile.close()
    # test stemming
    infile = open(DEL_STOP_WORD_CACHE, 'r')
    outfile = open(STEM_CACHE, 'w')
    lines = infile.read()
    lines = lines.split('\n')
    # stem sentences of all lines
    sentences = porter_stemming(lines)
    for s in sentences:
        outfile.write(s + '\n')
    print('The stop-word-free sentences has been stemmed. ')
    print('The result is output in ' + STEM_CACHE)
    outfile.close()


'''
// main //
'''

if __name__ == "__main__":
    DOC_PATH = 'doc/unprocessed_data/d30045t/'
    DOCUMENTS = ('NYT19981125.0417',
                 'NYT19981125.0433',
                 # 'NYT19981126.0192',
                 # 'NYT19981127.0203',
                 # 'NYT19981127.0240',
                 # 'NYT19981127.0256',
                 # 'NYT19981127.0264',
                 # 'NYT19981127.0289',
                 # 'NYT19981127.0293',
                 # 'NYT19981129.0113',
                 )
    doc = DocProcess(DOC_PATH, DOCUMENTS)
    doc6 = doc.ori_doc[6]
    p_doc6 = doc.processed_doc[6]
    pass
