import os
from yapf.yapflib.yapf_api import FormatFile
import glob

files = []
for file in glob.glob('optimal_pytorch/*.py'):
    files.append(file)
for file in glob.glob('examples/*.py'):
    files.append(file)
for file in glob.glob('tests/*.py'):
    files.append(file)
for file in glob.glob('experimental/*.py'):
    files.append(file)

# Add  for recursively doing for every python file or if given a list
for ele in files:
    print(ele)
    FormatFile(ele, in_place=True)