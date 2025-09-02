'''
   Experiment 6: Program
   Training with restart and jump start
'''
import copy
import os, sys

# usage: python train.py 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = str( (int(sys.argv[1])+1) %2)

import torch

from collections import deque

# training_mod = {
#     "start_method": 's-jump',
#     # "start_method": 'nojump',
#     "restart": True,
#     "less_start": False,
#     # "play_method": "renew_children",
#     "play_method": "deep_play",
#
#     # "log_name": "RC_double",
#     "log_name": "test_dir",
#     "slow_transfer": False,
#     "depth": 3,
#     #DEPTH = 3 IS 2-PLY
#     "cut_size": 1024,
#     "batch_size": 1024,
#     "shuffle": False,
#     "load_weight": True,
#     "weight_number": 2000,
#     "return_node": True,
#     "testing_log": False,
#     "learning_rate": 0.001,
#     # "learning_rate": 0.0001,
#     # "learning_rate": 0.00001,
#     "greedy_move": False,
#     "simple_value": False,
#     "best_only": False,
#     #bestonly:
#     "actual_value": False,
#     "freeze": False,
#     "D3play_D1learn": False,
#     "proportion_evaluation": False,
#     "proportion_standardization": False,
#     "random_state": False,
#     "discount_factor": 1,
#     "Original_double_learning": True,
#     #normalized means make all value go to 0~1. reward will be true_reward*(1/average_score)
#     "normalized": False,
#     #average_number means how many games will be considered for calculating true_reward*(1/average_score)
#     "average_number": 200,
#     "flip_number": [0,1],
#     # "flip_number": None,
#     #flip_number = "Rotate counterclockwise + symmetrical about the y-axis."
#     "keep_B": False,
#     # renew model_B or keep it
#     "act_1": "deep",
#     # act_1 is only "deep" or "simple"
#     "act_2": "simple",
#     # act_2 is only "no" or "simple"
#     "triple_learning": False,
#     "renew_all": False,
#     # renew_all is to judge renew both A and B together
#     "half_move": False,
#     # half_move: static_value, dynamic_move.
#     "same_state": False,
#     "RC_end_AFstate": False,
#     "double_mode": False,
#     "different_weight": True,
#     # different_weight is very important in double learning!!!
#     "only_different": False,
#     "full_formula": False,
#     "double_gpu": False,
#     "alpha": 1,
#     "gamma": 1,
#     # only renew different node
#
#     # now,another learning is only okay for D3
#     # "another_name": "random",
#     "another_name": "D1",
#     "another_possibility": 1,
#
# }

import json

def load_parameters(file_path):
    with open(file_path, 'r') as file:
        parameters = json.load(file)
    return parameters

file_path = f'{sys.argv[1]}.json'
training_mod = load_parameters(file_path)

class MovingAverage:
    def __init__(self, size=training_mod["average_number"]):
        self.size = size
        self.queue = deque(maxlen=size)

    def add(self, value):
        self.queue.append(value)

    def average(self):
        return np.mean(self.queue) if self.queue else 0

device_number = int(sys.argv[3])
if( training_mod["double_gpu"]):
    another_device_number = 1 ^ device_number
else:
    another_device_number = device_number
device0 = torch.device(f"cuda:{device_number}")
device1 = torch.device(f"cuda:{another_device_number}")

import numpy as np
import sys, os, random, logging
# import queue, threading, concurrent.futures
import torch.multiprocessing as tmp
from multiprocessing import Queue
import time

sys.path.append('../common')
import Game2048
import deep_play
import double_deep_play, triple_deep_play
import rotation_flip_match as rfm

import platform, resource

if (platform.system().lower() == "linux"):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

seed = int(sys.argv[2])

numprocess = 4
if (training_mod["triple_learning"]):
    numprocess = 2

BATCH_SIZE = training_mod["batch_size"]
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
model = modeler.Model(learning_rate=training_mod["learning_rate"]).to(device0)
model_B = None

# model = model.to(device1)
if (training_mod["Original_double_learning"] or training_mod["triple_learning"]):
    model_B = modeler.Model(learning_rate=training_mod["learning_rate"]).to(device1)

model_generate = None

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
torch.multiprocessing.set_start_method('spawn', force=True)

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
queue2 = tmp.Queue(maxsize=10)
queue3 = tmp.Queue(maxsize=10)

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


def generator(threadID, queue: tmp.Queue, modelg, model_B, slow_control: tmp.Value, queue_B: tmp.Queue = None):
    ''' A function for generator process
    :param threadID:
    :param queue:
    :param modelg:
    :param model_B: In double/half_doouble learning, model_B is trained.
    :param slow_control:
    '''

    if not training_mod.get("state_percentage"):
        state_percentage = 1.0
    else:
        state_percentage = training_mod.get("state_percentage")

    if (training_mod["proportion_evaluation"]):
        if (training_mod["return_node"] == False):
            raise Exception(f"proportion_evaluation but not return_node")
    if (training_mod["best_only"]):
        if (training_mod["return_node"] == False):
            raise Exception(f"best_only but not return_node")


    average_queue = MovingAverage()


    for itr in range(200):
        average_queue.add(2500)


    arpha = training_mod["alpha"]
    gamma = training_mod["gamma"]

    while True:
        average_score = average_queue.average()
        state = makeInitState(jump, threadID)
        turn = 0
        states = []
        for restartcount in range(10000):
            lastboard = None
            lastnode = None

            after_node_lis = []

            while True:
                turn += 1
                child_node_lis = []
                child_node_lis_B = []
                root_node = None
                root_node_B = None
                random_B = False if random.random() > 0.5 else True
                random_another = random.random()
                ev = 0
                if (training_mod["play_method"] == "simple_play"):
                    if(training_mod["normalized"]):
                        dir, ev = playalg.simplePlay(state, modelg, training_mod["normalized"], average_score)
                    else:
                        dir, ev = playalg.simplePlay(state, modelg)
                elif (training_mod["play_method"] == "deep_play"):

                    if (training_mod["Original_double_learning"]):
                        dir, dir_A, dir_B, ev_Ab, ev_Ba, root_node = double_deep_play.double_expand_and_get(state,
                                                                                                            modelg,
                                                                                                            model_B,
                                                                                                            int(
                                                                                                                training_mod[
                                                                                                                    "depth"]),
                                                                                                            greedy_value=
                                                                                                            training_mod[
                                                                                                                "simple_value"],
                                                                                                            return_node=True,
                                                                                                            greedy_move=
                                                                                                            training_mod[
                                                                                                                "greedy_move"],
                                                                                                            discount_factor=
                                                                                                            training_mod[
                                                                                                                "discount_factor"],
                                                                                                            double_learning=True,
                                                                                                            act_1=
                                                                                                            training_mod[
                                                                                                                "act_1"],
                                                                                                            act_2=
                                                                                                            training_mod[
                                                                                                                "act_2"],
                                                                                                            flip_number=
                                                                                                            training_mod[
                                                                                                                "flip_number"],
                                                                                                            device_0_number=device_number,
                                                                                                            device_1_number=another_device_number,
                                                                                                            normalized = training_mod["normalized"],
                                                                                                            average_score = average_score,

                                                                                                            )

                    else:
                            dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                          greedy_value=training_mod["simple_value"],
                                                                          return_node=True,
                                                                          greedy_move=training_mod["greedy_move"],
                                                                          discount_factor=training_mod[
                                                                              "discount_factor"],
                                                                          device_number=device_number,
                                                                          average_score=average_score,
                                                                          state_percentage=state_percentage,

                                                                          )
                            if (training_mod["D3play_D1learn"]):
                                dir1, ev1, root_node1 = deep_play.expand_and_get(state, modelg, 1,
                                                                                 greedy_value=training_mod[
                                                                                     "simple_value"], return_node=True,
                                                                                 greedy_move=training_mod[
                                                                                     "greedy_move"],
                                                                                 discount_factor=training_mod[
                                                                                     "discount_factor"],
                                                                                 average_score=average_score,
                                                                                 state_percentage=state_percentage,

                                                                                 )

                    child_node_lis = root_node.children

                else:
                    # elif (training_mod["play_method"] == "renew_children"):
                    if (training_mod["Original_double_learning"]):

                        dir, dir_A, dir_B, ev_Ab, ev_Ba, root_node = double_deep_play.double_expand_and_get(state,
                                                                                                            modelg,
                                                                                                            model_B,
                                                                                                            int(
                                                                                                                training_mod[
                                                                                                                    "depth"]),
                                                                                                            greedy_value=
                                                                                                            training_mod[
                                                                                                                "simple_value"],
                                                                                                            return_node=True,
                                                                                                            greedy_move=
                                                                                                            training_mod[
                                                                                                                "greedy_move"],
                                                                                                            discount_factor=
                                                                                                            training_mod[
                                                                                                                "discount_factor"],
                                                                                                            double_learning=True,
                                                                                                            renew_children=True,
                                                                                                            act_1=
                                                                                                            training_mod[
                                                                                                                "act_1"],
                                                                                                            act_2=
                                                                                                            training_mod[
                                                                                                                "act_2"],
                                                                                                            flip_number=
                                                                                                            training_mod[
                                                                                                                "flip_number"],
                                                                                                            device_0_number=device_number,
                                                                                                            device_1_number=another_device_number,
                                                                                                            normalized=
                                                                                                            training_mod[
                                                                                                                "normalized"],
                                                                                                            average_score=average_score,

                                                                                                            )
                        child_node_lis = root_node.children
                    else:
                        RC = True
                        if(training_mod.get("best_only")):
                            RC = False
                        dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                      return_node=True,
                                                                      greedy_move=training_mod["greedy_move"],
                                                                      random_state=training_mod["random_state"],
                                                                      discount_factor=training_mod["discount_factor"],
                                                                      device_number=device_number,
                                                                      average_score=average_score,
                                                                      RC = RC,
                                                                      state_percentage=state_percentage,

                                                                      )
                        child_node_lis = root_node.children

                # Play and Log

                if (training_mod["return_node"] == True):
                    if (training_mod["testing_log"]):

                        if (training_mod["Original_double_learning"]):
                            testing_logger.info(
                                f" process_number {threadID} restart {restartcount} turn {turn} double_evaluation {root_node.exp_A_B} evaluationA_b{root_node.exp_Ab} evaluationB_a{root_node.exp_Ba} score {state.score} best_move {root_node.next_move_A_B} dir {dir} possible_move {root_node.available_actions}"
                            )
                        else:
                            testing_logger.info(
                                f" process_number {threadID} restart {restartcount} turn {turn} evaluation {root_node.exp} score {state.score} best_move {root_node.next_move} dir {dir} possible_move {root_node.available_actions}")
                        testing_logger.info(
                            f" board {state.board}"
                        )
                        # child_number = 0
                        # for act in range(0, 4):
                        #     if (state.canMoveTo(act) == False):
                        #         testing_logger.info(
                        #             f" process_number {threadID} move {act} None"
                        #         )
                        #     else:
                        #         testing_logger.info(
                        #             f" process_number {threadID} move {act} evaluation {root_node.children[child_number].exp} value {root_node.children[child_number].value} score {root_node.children[child_number].state.score}"
                        #         )
                        #         child_number += 1

                state.play(dir)

                # Put data into queue
                if (training_mod["actual_value"] == False):
                    if (training_mod["play_method"] == "renew_children"):
                        if (training_mod["best_only"]):
                            if (training_mod["proportion_evaluation"] == True):
                                raise Exception(f"proportion_evaluation and best_only together")
                            queue.put({'lastboard': root_node.next_node.state.board, 'target': root_node.next_node.exp})
                        elif (training_mod["Original_double_learning"]):
                            random_number = random.random()
                            if (training_mod["renew_all"]):
                                if (training_mod["flip_number"] is None):
                                    for child_node in child_node_lis:
                                        queue.put(
                                            {'lastboard': child_node.state.board, 'target': child_node.exp_Ba})
                                    if(training_mod["keep_B"] == False):
                                        for child_node in child_node_lis:
                                            queue_B.put(
                                                {'lastboard': child_node.state.board, 'target': child_node.exp_Ab})
                                else:
                                    for child_node in child_node_lis:
                                        queue.put(
                                            {'lastboard': child_node.state.board, 'target': child_node.exp_Ba})
                                    if(training_mod["keep_B"] == False):
                                        for child_node in child_node_lis:
                                            queue_B.put({'lastboard': rfm.transform_2d(child_node.state.board,
                                                                                       training_mod["flip_number"][0],
                                                                                       training_mod["flip_number"][1]),
                                                         'target': child_node.exp_Ab})

                            else:
                                if(training_mod["flip_number"] is None):
                                    if (random_number < 0.5) or (training_mod["keep_B"] == True):
                                        for child_node in child_node_lis:
                                            queue.put({'lastboard': child_node.state.board, 'target': child_node.exp_Ba})
                                    else:
                                        for child_node in child_node_lis:
                                            queue_B.put({'lastboard': child_node.state.board, 'target': child_node.exp_Ab})
                                else:
                                    if (random_number < 0.5) or (training_mod["keep_B"] == True):
                                        for child_node in child_node_lis:
                                            queue.put({'lastboard': child_node.state.board, 'target': child_node.exp_Ba})
                                    else:
                                        for child_node in child_node_lis:
                                            queue_B.put({'lastboard': rfm.transform_2d(child_node.state.board, training_mod["flip_number"][0], training_mod["flip_number"][1]) , 'target': child_node.exp_Ab})
                        else:
                            for child_node in child_node_lis:
                                queue.put({'lastboard': child_node.state.board, 'target': child_node.exp})
                    else:
                        # normal renew (D3)
                        if lastboard is not None:

                            if (training_mod["Original_double_learning"]):
                                if (training_mod["full_formula"]):
                                    Node_AS_a = root_node.next_node_A
                                    r_Ba = (Node_AS_a.state.score - root_node.state.score)
                                    target_value_A = lastnode.value_A + arpha * (r_Ba + gamma * Node_AS_a.exp_Ba - lastnode.value_A)


                                    Node_AS_b = root_node.next_node_B
                                    r_Ab = (Node_AS_b.state.score - root_node.state.score)
                                    target_value_B = lastnode.value_B + arpha * (r_Ab + gamma * Node_AS_b.exp_Ab - lastnode.value_B)

                                if (training_mod["renew_all"]):
                                    if (training_mod["full_formula"]):
                                        queue.put({'lastboard': lastboard, 'target': target_value_A})
                                        if (training_mod["keep_B"] == False):
                                            if (training_mod["flip_number"] is None):
                                                queue_B.put({'lastboard': lastboard, 'target': target_value_B})
                                            else:
                                                queue_B.put({'lastboard': rfm.transform_2d(lastboard,training_mod["flip_number"][0], training_mod["flip_number"][1]), 'target': target_value_B})

                                    else:
                                        queue.put({'lastboard': lastboard, 'target': ev_Ba})
                                        if (training_mod["keep_B"] == False):
                                            if(training_mod["flip_number"] is None):
                                                queue_B.put({'lastboard': lastboard, 'target': ev_Ab})
                                            else:
                                                queue_B.put({'lastboard': rfm.transform_2d(lastboard, training_mod["flip_number"][0], training_mod["flip_number"][1]), 'target': ev_Ab})

                                else:
                                    random_number = random.random()
                                    if (random_number < 0.5) or (training_mod["keep_B"] == True):
                                        if(training_mod["full_formula"]):
                                            queue.put({'lastboard': lastboard, 'target': target_value_A})
                                        else:
                                            queue.put({'lastboard': lastboard, 'target': ev_Ba})
                                    else:
                                        if(training_mod["flip_number"] is None):
                                            if training_mod["full_formula"]:
                                                queue_B.put({'lastboard': lastboard, 'target': target_value_B})
                                            else:
                                                queue_B.put({'lastboard': lastboard, 'target': ev_Ab})
                                        else:
                                            if training_mod["full_formula"]:
                                                queue_B.put({'lastboard': rfm.transform_2d(lastboard,training_mod["flip_number"][0],training_mod["flip_number"][1]), 'target': target_value_B})
                                            else:
                                                queue_B.put({'lastboard': rfm.transform_2d(lastboard,training_mod["flip_number"][0],training_mod["flip_number"][1]), 'target': ev_Ab})
                            elif (training_mod["only_different"]):
                                if (dir1 != dir):
                                    queue.put({'lastboard': lastboard, 'target': ev})
                            else:
                                queue.put({'lastboard': lastboard, 'target': ev})

                if not (training_mod["same_state"]):
                    lastboard = state.clone().board
                    if(training_mod["full_formula"]):
                        lastnode = copy.deepcopy(root_node)
                    state.putNewTile()
                    states.append(state.clone())
                else:
                    lastboard = root_node.next_node.state.board
                    state = root_node.next_node.children[0].state.clone()
                    states.append(state.clone())

                if state.isGameOver():
                    if (training_mod["RC_end_AFstate"] or training_mod["play_method"] != "renew_children"):

                        if (training_mod["actual_value"] == False):
                            if (training_mod["Original_double_learning"]):
                                if (training_mod["renew_all"]):
                                    queue.put({'lastboard': lastboard, 'target': 0})
                                    if(training_mod["keep_B"] == False):
                                        queue_B.put({'lastboard': lastboard, 'target': 0})
                                else:
                                    random_number = random.random()
                                    if (random_number < 0.5):
                                        queue.put({'lastboard': lastboard, 'target': 0})
                                    else:
                                        if (training_mod["keep_B"] == False):
                                            queue_B.put({'lastboard': lastboard, 'target': 0})
                            else:
                                queue.put({'lastboard': lastboard, 'target': 0.0})

                    logger.info(
                        f'game over thread {threadID} restart {restartcount} score {state.score} length {len(states)}')
                    break

            average_queue.add(states[-1].score)
            if (training_mod["less_start"] == True):
                if (restart == False):
                    raise Exception(f"Less_start but not restart")

            if restart:
                # restart
                if (training_mod["less_start"]):
                    if (len(states) > 100):
                        state = states[len(states) // 2]
                        turn -= len(states) // 2
                        states = []
                    else:
                        # go back to game start
                        break
                else:
                    if (len(states) > 10):
                        state = states[len(states) // 2]
                        turn -= len(states) // 2
                        states = []
                    else:
                        # go back to game start
                        break
            else:
                break


def trainer(queue, modelt, model_B, slow_control: tmp.Value, training_device_number=None, model_name = None):
    ''' A function for trainer thread
    :param model_name:
    :param queue:
    :param modelt: We train modelt in half_double.
    :param model_B: Now, in step learning, we set model_B to be trained.
    :param slow_control:
    '''
    OUTPUT_TIMING = 1000000
    learned_actions = 0
    data_number = 0
    cut_size = training_mod["cut_size"]

    current_step = 0

    while True:
        trainrecords = []
        trainrecords_B = []
        while True:

            queue_data = queue.get()
            data_number = data_number + 1
            # print("queue_data:",data_number,queue_data,queue.qsize())
            trainrecords.append(queue_data)
            if len(trainrecords) >= BATCH_SIZE:
                # logger.info(f'len(trainrecords): {len(trainrecords)}')
                if (training_mod["shuffle"]):
                    random.shuffle(trainrecords)
                break

        if (training_mod["Original_double_learning"] ):
            # only use modelt
            training_device = torch.device(f"cuda:{training_device_number}")
            x = np.zeros([len(trainrecords), modelt.DIM_I], dtype="float32")
            y = torch.zeros([len(trainrecords), modelt.DIM_O], dtype=torch.float32).to(training_device)
            for (i, playrecord) in enumerate(trainrecords):
                modelt.make_input(x[i, :], playrecord['lastboard'])
                y[i, :] = torch.tensor(playrecord['target']).to(training_device)
            for batch in range(0, int(len(trainrecords) / cut_size) + 1):
                start_itr = batch * cut_size
                if (start_itr >= (len(trainrecords))):
                    break
                end_itr = (batch + 1) * cut_size
                if (end_itr >= (len(trainrecords))):
                    end_itr = len(trainrecords)
                if(training_mod.get("lock")==True):
                    modelt.train_mode(x_train=x[start_itr:end_itr], y_train=y[start_itr:end_itr], batch_size=cut_size,
                                      device_number=training_device_number)
                else:
                    modelt.train_mode(x_train=x[start_itr:end_itr], y_train=y[start_itr:end_itr], batch_size=cut_size,
                                      device_number=training_device_number)
            learned_actions += len(trainrecords)

            # logger.info(f'learned: {learned_actions:8,d} loss: {l:8,.2f}')
            logger.info(f'learned: {learned_actions:8,d}')

            # save parameters
            if ((learned_actions - len(trainrecords)) // OUTPUT_TIMING) < (learned_actions // OUTPUT_TIMING):
                if(training_mod["load_weight"]):
                    name_for_save = "./" + logdir + f"/weights-{model_name}-" + str(
                        learned_actions // OUTPUT_TIMING + training_mod["weight_number"])
                    torch.save(modelt.state_dict(), name_for_save)
                    logger.info(
                        f'saved parameters ./{logdir}/weights-{model_name}-{learned_actions // OUTPUT_TIMING + training_mod["weight_number"]}.*')
                else:
                    name_for_save = "./" + logdir + f"/weights-{model_name}-" + str(
                        learned_actions // OUTPUT_TIMING)
                    torch.save(modelt.state_dict(), name_for_save)
                    logger.info(
                        f'saved parameters ./{logdir}/weights-{model_name}-{learned_actions // OUTPUT_TIMING}.*')
        else:
            x = np.zeros([len(trainrecords), modelt.DIM_I], dtype="float32")
            y = torch.zeros([len(trainrecords), modelt.DIM_O], dtype=torch.float32).to(device0)
            for (i, playrecord) in enumerate(trainrecords):
                modelt.make_input(x[i, :], playrecord['lastboard'])
                y[i, :] = torch.tensor(playrecord['target']).to(device0)
            for batch in range(0, int(len(trainrecords) / cut_size) + 1):
                start_itr = batch * cut_size
                if (start_itr >= (len(trainrecords))):
                    break
                end_itr = (batch + 1) * cut_size
                if (end_itr >= (len(trainrecords))):
                    end_itr = len(trainrecords)
                modelt.train_mode(x_train=x[start_itr:end_itr], y_train=y[start_itr:end_itr], batch_size=cut_size,
                                  device_number=device_number)
            learned_actions += len(trainrecords)

            # logger.info(f'learned: {learned_actions:8,d} loss: {l:8,.2f}')
            logger.info(f'model_{training_device_number}_learned: {learned_actions:8,d}')

            # output parameters
            if ((learned_actions - len(trainrecords)) // OUTPUT_TIMING) < (learned_actions // OUTPUT_TIMING):
                if(training_mod["load_weight"]):
                    name_for_save = "./" + logdir + "/weights-" + str(learned_actions // OUTPUT_TIMING + training_mod["weight_number"])
                    torch.save(modelt.state_dict(), name_for_save)
                    logger.info(f'saved parameters ./{logdir}/weights-{learned_actions // OUTPUT_TIMING + training_mod["weight_number"]}.*')
                else:
                    name_for_save = "./" + logdir + "/weights-" + str(learned_actions // OUTPUT_TIMING)
                    torch.save(modelt.state_dict(), name_for_save)
                    logger.info(f'saved parameters ./{logdir}/weights-{learned_actions // OUTPUT_TIMING}.*')




# generator_executor = concurrent.futures.ThreadPoolExecutor(max_workers=numprocess)
# for  i in range(numprocess): generator_executor.submit(generator(i))
# # generator(0)()
#
# print("start training")
# trainer()


if __name__ == '__main__':
    processes = []

    if (training_mod["load_weight"] == True):

        if (training_mod["double_mode"] == False):
            checkpointprefix = f'{logdir}/weights-{training_mod["weight_number"]}'
            model.load_state_dict(torch.load(checkpointprefix, map_location=device0))
        else:
            checkpointprefix = f'{logdir}/weights-another'
            model.load_state_dict(torch.load(checkpointprefix, map_location=device0))


        if (training_mod["Original_double_learning"]):
            if (training_mod["different_weight"]):
                checkpointprefix = f'{logdir}/weights-another'
            model_B.load_state_dict(torch.load(checkpointprefix, map_location=device1))

    if (training_mod["freeze"]):
        modeler.freeze(model)
        if (training_mod["Original_double_learning"]):
            modeler.freeze(model_B)

    model.share_memory()
    if (training_mod["Original_double_learning"]):
        model_B.share_memory()
    slow_control = tmp.Value('d', 0.0)
    # now, slow control is useless

    for i in range(numprocess):
        if (training_mod["Original_double_learning"]):
            processes.append(
                tmp.Process(target=generator, args=(i, queue1, model, model_B, slow_control, queue2)))
        else:
            processes.append(tmp.Process(target=generator, args=(i, queue1, model, model_B, slow_control,)))
    for process in processes:
        process.start()

    if (training_mod["Original_double_learning"]):
        training_process1 = tmp.Process(target=trainer, args=(queue1, model, None, slow_control, device_number,"A"))
        if (training_mod["keep_B"] == False):
            training_process2 = tmp.Process(target=trainer,
                                             args=(queue2, model_B, None, slow_control, another_device_number,"B"))
        training_process1.start()
        if (training_mod["keep_B"] == False):
            training_process2.start()
    else:
        if (training_mod["keep_B"] == False):
            training_process = tmp.Process(target=trainer, args=(queue1, model, model_B, slow_control, device_number,"B"))
            training_process.start()

    for process in processes:
        process.join()

    if (training_mod["Original_double_learning"]):
        training_process1.join()
        if (training_mod["keep_B"] == False):
            training_process2.join()
    else:
        if (training_mod["keep_B"] == False):
            training_process.join()
