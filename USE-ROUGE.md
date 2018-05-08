run 'convert_format_pyrouge.py' 

under ROUGE directory, in terminal:

$ ./ROUGE-1.5.5.pl -a -e data -x -b 665 -m -n 4 -w 1.2 -u -c 95 -r 1000 -f A -p 0.5 -t 0 file:///home/wang/Documents/Pycharm/DocumentSummarization/rouge_test.xml 

$ ./ROUGE-1.5.5.pl -a -e data -x -b 665 -m -n 4 -w 1.2 -u -c 95 -r 1000 -f A -p 0.5 -t 0 file:///home/wang/Documents/Pycharm/DocumentSummarization/rouge_test.xml >Reinforced.txt

The result score is located in Reinforced.txt

