from pyrouge import Rouge155

'''
r = Rouge155()
r.system_dir = 'systems/04systems'
r.model_dir = 'model/04model'
r.system_filename_pattern = 'D30045.M.100.T.TT'
r.model_filename_pattern = 'D30045.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
'''

'''
convert text into rouge format without changing filename.
'''


def convert2rouge_format():
    model_input_dir = 'doc/model/04model'
    model_output_dir = 'doc/model/04model_rouge'
    system_input_dir = 'doc/systems/04systems'
    system_output_dir = 'doc/systems/04systems_rouge'

    Rouge155.convert_summaries_to_rouge_format(model_input_dir, model_output_dir)
    Rouge155.convert_summaries_to_rouge_format(system_input_dir, system_output_dir)


if __name__ == '__main__':
    convert2rouge_format()
