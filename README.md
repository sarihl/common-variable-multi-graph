# common-variable-multi-graph
A python implementation of the paper 'A Common Variable Minimax Theorem for Graphs'by Ronald R. Coifman et al - https://doi.org/10.1007/s10208-022-09558-8
all credit goes to the authors.

This implmentation makes use of pytorch for gpu acceleration if available, although a gpu is not needed.
note: the implmentation could be easily changed to work using numpy alone, as it does not make any use of torch exculsive functions.

The code includes three files:
  1. main.py: just makes the calls to the adequete functions.
  2. algorithm.py: the implmentation of the aglorithm
  3. experiments.py: I have recreated the first 3 expermints shown in the paper, alongside figures 1-9. the full implmentation is available here.

Note: this is NOT an official implmentation, I tried to stick as much as possible to the paper, but there might be some deviations that I am not aware of. I will be happy to fix them if you notify me.
