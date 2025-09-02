import os, sys


import queue

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str( (int(sys.argv[1])+1) %2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

import numpy as np
import sys, os, random, logging
# import queue, threading, concurrent.futures
import torch.multiprocessing as tmp
from multiprocessing import Queue
import time

sys.path.append('../common')
import Game2048
import deep_play
import playalg

import platform
if (platform.system().lower() == "linux"):
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))
torch.multiprocessing.set_start_method('spawn', force=True)
# torch.cuda.set_device(0)
# device = 0
seed = 1
numprocess = 1
import cnn22B as modeler
model = modeler.Model(learning_rate=0.00001).cuda()

queue1 = tmp.Queue(maxsize=10)
queue2 = tmp.Queue(maxsize=10)

def makeInitState():
    state = Game2048.State()
    state.initGame()
    return state


def generator(queue_simple: queue.Queue, queue_rc: queue.Queue, modelg: modeler.Model):
    checkpointprefix = f'./weights-1000'
    modelg.load_state_dict(torch.load(checkpointprefix))
    for game_number in range(6):
        state = makeInitState()
        turn = 0
        lastboard = None
        while True:
            turn += 1
            child_node_lis = []
            root_node = None
            # dir_simple, ev_simple = playalg.simplePlay(state, modelg)
            dir_simple, ev_simple = deep_play.expand_and_get(state, modelg, 1)

            dir_RC, ev_RC, root_node_RC = deep_play.expand_and_get(state, modelg, 3, return_node=True, greedy_move=True,
                                                                   random_state=True, discount_factor=1)
            child_node_lis = root_node_RC.children
            state.play(dir_simple)
            if lastboard is not None:
                queue_simple.put({'lastboard': lastboard, 'target': ev_simple})
            queue_rc.put({'lastboard': root_node_RC.next_node.state.board, 'target': root_node_RC.next_node.exp})

            # lastboard = state.clone().board
            # state.putNewTile()

            lastboard = root_node_RC.next_node.state.board
            state = root_node_RC.next_node.children[0].state.clone()

            # print(dir_RC,dir_simple)

            if state.isGameOver():
                queue_simple.put({'lastboard': lastboard, 'target': 0.0})
                # queue_rc.put({'lastboard': lastboard, 'target': 0.0})
                print(state.score,state.board)
                break


def geter(queue_datas:queue.Queue, file_number):
    file_name = f"out_{file_number}"
    out_file = open(file_name,"w",encoding="utf-8")
    while True:
        try:
            queue_data = queue_datas.get(timeout= 15)
        except:
            break
        out_line = str(queue_data["lastboard"])+" "+str(queue_data['target'])+"\n"
        out_file.write(out_line)

if __name__ == '__main__':
    # model.load_state_dict(torch.load("weights-1000"))
    checkpointprefix = f'./weights-1000'
    model.load_state_dict(torch.load(checkpointprefix))
    model.share_memory()
    processes= []
    processes.append(
        tmp.Process(target=generator, args=(queue1, queue2, model))
    )
    for process in processes:
        process.start()
    getting_process1 = tmp.Process(target=geter, args=(queue1, 1))
    getting_process2 = tmp.Process(target=geter, args=(queue2, 2))
    getting_process1.start()
    getting_process2.start()
    for process in processes:
        process.join()
    getting_process1.join()
    getting_process2.join()
