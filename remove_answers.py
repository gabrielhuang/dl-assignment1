'''
@brief Preprocess python script to remove answers
@author Gabriel HUANG
@date October 2015

usage: python remove_answers.py script.py > stripped.py

1. Remove all code between
# <HW>
and 
# </HW>
tags

2. Replace with
# <PUT YOUR CODE HERE>
tags
'''

BEGIN = '#<HW>'
END = '#</HW>'
REPLACEMENT = [
'#',
'# **********************',
'# * PUT YOUR CODE HERE *',
'# **********************',
'#']


import sys

def get_prefix(line):
    start = line.find('#')
    prefix = line[:start]
    return prefix

def get_replacement(prefix):
    repl = [prefix + r for r in REPLACEMENT]
    return '\n'.join(repl)
    
try:
    f = open(sys.argv[1])
except:
    print 'Usage: {} script.py > stripped.py\n'.format(sys.argv[0])
    raise
    
    
in_hw = False
for line_number, line in enumerate(f):
    no_space = ''.join(line.split())
    if no_space == BEGIN:
        if in_hw:
            raise Exception('LINE {}: {}\nUnexpected BEGIN'.format(line_number, line) )
        else:
            in_hw = True
            prefix = get_prefix(line)
    elif no_space == END:
        if not in_hw:
            raise Exception('LINE {}: {}\nUnexpected END'.format(line_number, line))
        else:
            in_hw = False
            sys.stdout.write(get_replacement(prefix))
    elif not in_hw:
        sys.stdout.write(line)