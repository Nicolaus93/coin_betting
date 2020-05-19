from yapf.yapflib.yapf_api import FormatFile

# Add  for recursively doing for every python file or if given a list
FormatFile("examples/mnist.py", in_place=True)