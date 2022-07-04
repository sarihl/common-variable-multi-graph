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

The recreated figures of the first experiment:
![image](https://user-images.githubusercontent.com/57064636/177176548-f9129c2b-1fe0-4b94-a7c0-d0123424ed9b.png)
![image](https://user-images.githubusercontent.com/57064636/177176568-d0f28713-d9cb-4dbd-aada-08e3bcd9c700.png)
and the main result!
![image](https://user-images.githubusercontent.com/57064636/177176590-51ee9a1c-8f62-4d85-9f57-0c61b191f8fe.png)
