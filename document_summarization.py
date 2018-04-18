from poter_stemming import PorterStemmer

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
SPLIT_CACHE = 'doc/cache/split.cache'
DEL_STOP_WORD_CACHE = 'doc/cache/del_stop_word.cache'
STEM_CACHE = 'doc/cache/stem.cache'

'''
def split_sentences():
    outfile = open(PROCESS_CACHE, 'w')
    doc_index = 0
    for f in DOCUMENTS:
        infile = open(DOC_PATH + f, 'r')
        
        # output <index> to indict the beginning of the current document
        outfile.write('<' + str(doc_index) + '>\n')
        sentence = ''
        # if there is '.' the flag are to be set
        # the flag makes sentence is not split until
        # the next letter is big case, which is supposed
        # to be the signal of a new sentence
        sentence_flag = False
        
        while infile.readline() != '<TEXT>\n':
            continue
        while True:
            line = infile.readline()
            # if not the end of the text
            if line != '</TEXT>\n':
                # check every character in line
                for char in line:
                    if char == '\n':
                        continue
                    # ignore '``'
                    if char == '`':
                        continue
                    # ignore '\'\'' but the shortcoming is
                    # the '\'' such as 'it's' will become 'its'
                    if char == '\'':
                        continue
                    elif char == '.':
                        if sentence_flag:
                            # delete the space which appears
                            # in the front of the sentence
                            if sentence[0] == ' ':
                                sentence = sentence[1:]
                            outfile.write(sentence + '\n')
                            sentence = ''
                        else:  # sentence_flag unset
                            sentence_flag = True
                    else:
                        sentence += char
            # if the end of the text, switch to the next document
            else:  # line == '</TEXT>\n'
                # output <\index> to indict the end of the current document
                outfile.write('</' + str(doc_index) + '>\n')
                doc_index += 1
                break
    outfile.close()
'''

'''
Split text in the document into sentences.return a list, which contains sub-lists 
of sentences in a document, of all the text of the documents in the DOCUMENTS tuple.  
NOTE that there are some sentences are not to be recognized such as ones end with '?' 
or end with '.\'\'', as well as some extra punctuation such as `` or '' are remained
too.  
'''


def split_sentences():
    doc_sentences = []  # a list to store all sentences in every document
    # transverse every document
    for f in DOCUMENTS:
        # open the current document
        infile = open(DOC_PATH + f, 'r')
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
        doc_sentences.append(sentences)
        infile.close()
    # / for f in DOCUMENTS
    # since all sentences which are saved as sub-lists in doc_sentences
    # has been simply split, now the doc_sentences
    return doc_sentences


'''
delete stop word in sentence as well as transform the sentence into 
lower case 
'''


def delete_stop_words(sentence: str):
    # for identifying the real stop word
    # from sub-string in a word which is as same
    # as some stop word, it is more simple to add
    # space in the front and at the end of the sentence,
    # while using ' ' + word + ' ' to identify the stop
    # word
    sentence = ' ' + sentence.lower() + ' '
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


'''
//  main  //
'''


if __name__ == '__main__':
    # test split sentences
    outfile = open(SPLIT_CACHE, 'w')
    doc_index = 0
    doc_sentences = split_sentences()
    for sentences in doc_sentences:
        # output <index> to indict the beginning of the current document
        outfile.write('<' + str(doc_index) + '>\n')
        for s in sentences:
            outfile.write(s + '\n')
        # output <\index> to indict the end of the current document
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
    sentences = porter_stemming(lines)
    for s in sentences:
        outfile.write(s + '\n')
    print('The stop-word-free sentences has been stemmed. ')
    print('The result is output in ' + STEM_CACHE)
    outfile.close()
