import numpy as np
import sys
# test for training log
logdir = f'test_deep_play'
logfile_name = f'{logdir}/deep_play.log'
logfile = open(logfile_name,"r",encoding="utf-8")
lines = logfile.readlines()
number_lis =[]
for line in lines:
    if(line.__contains__("GameOver")==False):
        continue
    # print(line.split(" ")[2])
    number = int(line.split(" ")[2])
    number_lis.append(number)
print(sum(number_lis)/len(number_lis))
