'''
   Experiment 6: Program
   Training with restart and jump start
'''

import os,sys
# usage: python train.py 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str( (int(sys.argv[1])+1) %2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
training_mod = {
    "start_method": 's-jump',
    "restart" : True,
    # "play_method": "deep_play",
    "play_method": "deep_play",
    # "play_method": "test_renew_children_play",
    # "play_method": "renew_children",
    # "play_method": "greedy_move",
    # test_renew_children_play : renew_children with return children_node
    # "play_method": "renew_children",
    # "play_method": "simple_play",
    # "log_name": "deep_move_greedy_value_from1000",
    # "log_name": "d3_lowrate_from1000",
    # RCS: renew_children_separate with separate model
    # "log_name": "RCS_from_0",
    # "log_name": "RCS",
    # "log_name": "RCS_nostart",

    # "log_name": "RCS_from_0_nostart",
    # "log_name": "RCRS_nostart",
    # "log_name": "real_deep_1",
    # "log_name": "large_batch",
    "log_name": "RCR",
    # "log_name": "test_deep_3",
    # "simple_play": T,
    "depth": int(3),
    "simple_value":False,
    "cut_size": 16384,
    "load_1000": True,
    "return_node": True,
    "separate_mod": False,
    "actual_value": True,
    "testing_log": False,
}


import numpy as np
import sys, os, random, logging
# import queue, threading, concurrent.futures
import torch.multiprocessing as tmp
from multiprocessing import Queue
import time

sys.path.append('../common')
import Game2048
import deep_play

import platform,resource
if(platform.system().lower() == "linux"):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(100000,rlimit[1]))


# torch.cuda.set_device(0)
device = 0

seed = int(sys.argv[1])

numprocess = 4
BATCH_SIZE = 16384
modelname = 'cnn22B'
restart = training_mod["restart"]
jump = training_mod["start_method"]
# SEARCH_HEIGHT = 3
# Logging
logdir0 = f'{training_mod["log_name"]}'
if not os.path.exists(logdir0): os.makedirs(logdir0)
logdir = f'{training_mod["log_name"]}/training_{modelname}_{seed}'
if not os.path.exists(logdir): os.makedirs(logdir)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{logdir}/training.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger.addHandler(fh)
logger.info(f'Execution parameters: {sys.argv}')

# Preparing model and seed
exec(f'import {modelname} as modeler')
model = modeler.Model(learning_rate = 0.00001).cuda()
model_generate = None
if(training_mod["separate_mod"]):
    model_generate = modeler.Model(learning_rate = 0.00001).cuda()

logger.info(f'NN model: {model.description}')
logger.info(f'NN number of parameters: {model.numparams}')

testing_logger = logging.getLogger("testing_logger")
testing_logger.setLevel(logging.INFO)
fh_test = logging.FileHandler(f'{logdir}/testing.log')
fh_test.setLevel(logging.INFO)
fh_test.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
testing_logger.addHandler(fh_test)
testing_logger.info(f'Execution parameters: {sys.argv}')

random.seed(seed)
np.random.seed(seed)
# tc.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.multiprocessing.set_start_method('spawn',force=True)


# tf.random.set_seed(seed)
np.set_printoptions(threshold=np.inf)

# Preparing session
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver(max_to_keep=None)

# mg = tmp.Manager()

# queue1 = mg.Queue(maxsize=50)
queue1 = tmp.Queue(maxsize=10)
# queue1 = Queue(maxsize=10)

import playalg


def makeInitState(jump, threadID):
    if jump == 'nojump':
        state = Game2048.State()
        state.initGame()
        return state
    elif jump == 's-jump':
        cells = list(range(16));
        random.shuffle(cells)
        tiles = [[1 if random.random() < 0.9 else 2, 1 if random.random() < 0.9 else 2],
                 [11, 1 if random.random() < 0.9 else 2],
                 [12, 1 if random.random() < 0.9 else 2],
                 [11, 12],
                 [13, 1 if random.random() < 0.9 else 2]]
        state = Game2048.State()
        state.board[cells[0]] = tiles[threadID][0]
        state.board[cells[1]] = tiles[threadID][1]
        return state
    elif jump == 'l-jump':
        cells = list(range(16));
        random.shuffle(cells)
        tiles = [[1 if random.random() < 0.9 else 2, 1 if random.random() < 0.9 else 2],
                 [12, 1 if random.random() < 0.9 else 2],
                 [13, 1 if random.random() < 0.9 else 2],
                 [12, 13],
                 [14, 1 if random.random() < 0.9 else 2]]
        state = Game2048.State()
        state.board[cells[0]] = tiles[threadID][0]
        state.board[cells[1]] = tiles[threadID][1]
        return state
    else:
        raise ValueError(jump)

def generator(threadID,queue,modelg):
    ''' A function for generator process '''

    while True:
        state = makeInitState(jump, threadID)
        turn = 0
        states = []
        for restartcount in range(10000):
            lastboard = None
            after_node_lis = []

            while True:
                turn += 1
                child_node_lis = []
                root_node = None
                ev = 0
                if(training_mod["play_method"] == "simple_play"):
                    dir, ev = playalg.simplePlay(state, modelg)
                elif (training_mod["play_method"] == "deep_play"):
                    if(training_mod["return_node"] == False):
                        dir, ev = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                           greedy_value=training_mod["simple_value"])
                    else:
                        dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                      greedy_value=training_mod["simple_value"],
                                                                      return_node=True)
                        child_node_lis = root_node.children
                elif (training_mod["play_method"] == "test_renew_children_play"):
                    dir,ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                 return_node=True)
                    child_node_lis = root_node.children
                elif (training_mod["play_method"] == "greedy_move"):
                    dir,ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                 return_node=True, greedy_move=True)
                    child_node_lis = root_node.children
                else:
                # elif (training_mod["play_method"] == "renew_children"):
                    dir,ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                 return_node=True)
                    child_node_lis = root_node.children

                if(training_mod["actual_value"]):
                    if (training_mod["return_node"] == False):
                        raise Exception(f"Want actual_node but didnot return node")
                    after_node_lis.append(root_node)

                if(training_mod["return_node"] == True):
                    if(training_mod["testing_log"]):
                        testing_logger.info(
                            f" process_number {threadID} restart {restartcount} turn {turn} evaluation {root_node.exp} score {state.score} best_move {root_node.next_move} possible_move {root_node.available_actions}")
                        testing_logger.info(
                            f" board {state.board}"
                        )
                        child_number = 0
                        for act in range(0,4):
                            if(state.canMoveTo(act) == False):
                                testing_logger.info(
                                    f" process_number {threadID} move {act} None"
                                )
                            else:
                                testing_logger.info(
                                    f" process_number {threadID} move {act} evaluation {root_node.children[child_number].exp} value {root_node.children[child_number].value} score {root_node.children[child_number].state.score}"
                                )
                                child_number += 1

                state.play(dir)

                if(training_mod["actual_value"] == False):
                    if (training_mod["play_method"] == "renew_children" or training_mod["play_method"] == "test_renew_children_play"):
                        for child_node in child_node_lis:
                            queue.put({'lastboard': child_node.state.board, 'target': child_node.exp})
                    else:
                        if lastboard is not None:
                            queue.put( {'lastboard': lastboard, 'target': ev} )

                lastboard = state.clone().board
                state.putNewTile()
                states.append(state.clone())
                if state.isGameOver():
                    if (training_mod["actual_value"] == False):
                        queue.put( {'lastboard':lastboard, 'target':0.0} )
                    logger.info(f'game over thread {threadID} restart {restartcount} score {state.score} length {len(states)}')
                    break

            if (training_mod["actual_value"]):
                if (after_node_lis == []):
                    raise Exception(f"after_node_lis is empty")
                after_node_lenth = len(after_node_lis)
                after_node_lis[-1].real_target = 0
                for nod_itr in range(2,after_node_lenth+1):
                    reward = after_node_lis[-nod_itr+1].state.score - after_node_lis[-nod_itr].state.score
                    after_node_lis[-nod_itr].real_target = after_node_lis[-nod_itr+1].real_target + reward
                for node in after_node_lis:
                    queue.put({'lastboard': node.state.board, 'target': node.real_target})


            if restart and len(states) > 10:
                # restart
                state = states[len(states)//2]; turn -= len(states)//2; states = []
            else:
                # go back to game start
                break


def trainer(queue,modelt):
    ''' A function for trainer thread '''
    OUTPUT_TIMING = 1000000
    learned_actions = 0
    data_number = 0
    cut_size = training_mod["cut_size"]
    while True:
        trainrecords = []
        while True:
            queue_data = queue.get()
            data_number = data_number+1
            # print("queue_data:",data_number,queue_data,queue.qsize())
            trainrecords.append(queue_data)
            if len(trainrecords) >= BATCH_SIZE: break

        x = np.zeros([len(trainrecords), modelt.DIM_I],dtype="float32")
        # y = np.zeros([len(trainrecords), modelt.DIM_O],dtype="float32")
        y = torch.zeros(  [len(trainrecords), modelt.DIM_O],dtype=torch.float32 ).cuda()
        for (i, playrecord) in enumerate(trainrecords):
            modelt.make_input(x[i,:], playrecord['lastboard'])
            # y[i,:] =  playrecord['target']
            y[i,:] =  torch.tensor(playrecord['target']).cuda()

        # l, _ = sess.run([modelt.loss, modelt.train_step], feed_dict={modelt.input:x, modelt.correct:y})

        for batch in range(0,int (len(trainrecords)/ cut_size) + 1):
            start_itr = batch * cut_size
            if( start_itr >= (len(trainrecords)) ):
                break
            end_itr = (batch+1) * cut_size
            if(end_itr >= (len(trainrecords))):
                end_itr = len(trainrecords)
            modelt.train_mode(x_train=x[start_itr:end_itr], y_train=y[start_itr:end_itr], batch_size=cut_size)
        learned_actions += len(trainrecords)

        # logger.info(f'learned: {learned_actions:8,d} loss: {l:8,.2f}')
        logger.info(f'learned: {learned_actions:8,d}')

        # output parameters
        if ((learned_actions - len(trainrecords)) // OUTPUT_TIMING) < (learned_actions // OUTPUT_TIMING):
            # saver.save(sess, f'./{logdir}/weights', global_step=(learned_actions//OUTPUT_TIMING))
            name_for_save = "./"+logdir+"/weights-"+str(learned_actions//OUTPUT_TIMING + 1100)
            # modelt.save_weight(weight_name=name_for_save)
            torch.save(modelt.state_dict(),name_for_save)
            logger.info(f'saved parameters ./{logdir}/weights-{learned_actions//OUTPUT_TIMING + 1100}.*')

# generator_executor = concurrent.futures.ThreadPoolExecutor(max_workers=numprocess)
# for  i in range(numprocess): generator_executor.submit(generator(i))
# # generator(0)()
#
# print("start training")
# trainer()


if __name__ == '__main__':
    processes = []
    if(training_mod["load_1000"] == True):
        checkpointprefix = f'{logdir }/weights-1100'
        model.load_state_dict(torch.load(checkpointprefix))
        if (training_mod["separate_mod"]):
            model_generate.load_state_dict(torch.load(checkpointprefix))

    model.share_memory()
    for i in range(numprocess):
        if (training_mod["separate_mod"]):
            processes.append(tmp.Process(target=generator, args=(i, queue1,model_generate,)))
        else:
            processes.append( tmp.Process(target=generator, args=(i,queue1,model,)) )
    for process in processes:
        process.start()
    training_process = tmp.Process(target=trainer, args=(queue1,model,))

    training_process.start()
    for process in processes:
        process.join()
    training_process.join()


