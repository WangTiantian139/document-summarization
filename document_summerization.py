DOC_PATH = 'doc/unprocessed_data/d30045t/'
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
PROCESS_CACHE = 'doc/cache/process.cache'


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


if __name__ == '__main__':
    split_sentences()
    print('The unprocessed data has been split into sentences by space. ')
    print('The result is output in ' + PROCESS_CACHE)
