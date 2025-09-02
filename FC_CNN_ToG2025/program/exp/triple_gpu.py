'''
   Experiment 6: Program
   Training with restart and jump start
'''

import os, sys

# usage: python train.py 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = str( (int(sys.argv[1])+1) %2)

import torch

training_mod = {
    "start_method": 's-jump',
    "restart": True,
    "less_start": False,
    # "play_method": "renew_children",
    "play_method": "deep_play",
    # "play_method": "test_renew_children_play",

    # "play_method": "greedy_move_deep_value",
    # test_renew_children_play : renew_children with return children_node
    # "play_method": "renew_children",
    # "play_method": "simple_play",
    # "log_name": "greedy_move_deep_value_less_start",
    # "log_name": "real_deep_1_nostart",
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
    # "log_name": "deep_play",
    # "log_name": "test_deep_3",
    # "log_name": "real_1_from0",
    # "log_name": "greedy_move_renew_children",
    # "log_name": "renew_children_from_0",
    # "log_name": "D3_shuffle_zero",
    # "log_name": "RC_shuffle_zero",
    # "log_name": "deep_3_shuffle",
    # "log_name": "deep_1_shuffle",
    # "log_name": "RC_shuffle_proportion",
    # "log_name": "RC_shuffle_proportion_std",
    # "log_name": "RC_shuffle_renew_best",
    # "log_name": "RC_shuffle_renew_best_randomS_simple_move",
    # "log_name": "RC_shuffle_renew_best_randomS_simple_move_same_state",
    # "log_name": "RC_shuffle_renew_best_simple_move_noend",
    # "log_name": "RC_shuffle_renew_best_simple_move_withend",
    # "log_name": "RC_shuffle_simple_move_4times",
    # "log_name": "RC_shuffle_simple_value_4times",
    # "log_name": "D3_shuffle_simple_move",
    # "log_name": "D3_shuffle_simple_value",
    # "log_name": "D3_greedy_move_greedy_value",
    # "log_name": "D3_shuffle_proportion",
    # "log_name": "D3_shuffle_proportion_std",
    # "log_name": "deep_move_simple_value_shuffle",
    # "log_name": "RC_shuffle_factor95",
    # "log_name": "D3_shuffle_factor95",
    # "log_name": "D3_shuffle_double",
    # "log_name": "RC_shuffle_double",
    # "log_name": "RC_shuffle",
    # "log_name": "D3_shuffle_simple_value",
    # "log_name": "D3_shuffle_double_simple_value",
    # "log_name": "D3_shuffle_half_double_2",
    # "log_name": "D3_shuffle_half_double_value",
    # "log_name": "D3_shuffle_half_double_value_simpleM",
    # "log_name": "D3_shuffle_half_double_move_simpleV",

    # "log_name": "D3_1K_2000step",
    # "log_name": "D3_1K_5000step",
    # "log_name": "D3_1K_10000step",
    # "log_name": "D3_1K_5000step_double_2",
    # "log_name": "D3_1K_5000step_double_opposite",
    # "log_name": "D3_1K_10000step_double_opposite",
    # "log_name": "D3_1K_500step_double",
    # "log_name": "D1_shuffle",
    # "log_name": "D3_shuffle",
    # "log_name": "D3_shuffle_1048k",
    # "log_name": "D1_nozero_fromzero",
    # "log_name": "D3_shuffle_half_double_another",
    # "log_name": "D1_shuffle_half_double_another",
    # "log_name": "D3_shuffle_PD1_100",
    # "log_name": "D3_shuffle_Prandom_10",
    # "log_name": "D3_shuffle_D3play_D1learn",
    # "log_name": "D3_shuffle_original_double_all",
    # "log_name": "D3_shuffle_original_double",
    # "log_name": "test",
    # "log_name": "D3_shuffle_freeze",
    # "log_name": "D3_shuffle_original_double_freeze",
    # "log_name": "D3_shuffle_original_double_another",
    # "log_name": "D1_shuffle_original_double",
    # "log_name": "D3_shuffle_original_double_from0",
    # "log_name": "D3_shuffle_only_different",
    "log_name": "D3_shuffle_triple_learning",
    "slow_transfer": False,
    "depth": 3,
    # "cut_size": 1024,
    # "batch_size": 1024,
    "cut_size": 16384,
    # "batch_size": 16384,
    "batch_size": 131072,
    # "batch_size" : 1048576,
    "load_1000": True,
    "return_node": True,
    "separate_mod": False,
    "actual_value": False,
    "testing_log": True,
    # "learning_rate": 0.001,
    "learning_rate": 0.00001,
    "greedy_move": False,
    "simple_value": False,
    "shuffle": False,
    "freeze": False,
    "D3play_D1learn": False,
    "proportion_evaluation": False,
    "proportion_standardization": False,
    "best_only": False,
    "random_state": False,
    "discount_factor": 1,
    "Double_learning": False,
    "Original_double_learning": False,
    "triple_learning": True,
    "renew_all": False,
    # renew_all is to judge renew both A and B together
    "half_double_learning": False,
    "half_move": False,
    "same_state": False,
    "RC_end_AFstate": False,
    "step_learning": False,
    "step": 5000,
    "double_opposite": False,
    "double_mode": False,
    "another_learning": False,
    "different_weight": True,
    "only_different": False,
    # only renew different node

    # now,another learning is only okay for D3
    # "another_name": "random",
    "another_name": "D1",
    "another_possibility": 1,

}
device_number = int(sys.argv[2])
another_device_number = 1 ^ device_number
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

import platform, resource

if (platform.system().lower() == "linux"):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

seed = int(sys.argv[1])

numprocess = 4
if(training_mod["triple_learning"]):
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
model_C = None
# model = model.to(device1)
if (training_mod["Double_learning"] or training_mod["half_double_learning"] or training_mod["step_learning"] or
        training_mod["Original_double_learning"] or training_mod["triple_learning"]):
    model_B = modeler.Model(learning_rate=training_mod["learning_rate"]).to(device1)
if(training_mod["triple_learning"]):
    model_C = modeler.Model(learning_rate=training_mod["learning_rate"]).to(device1)
model_generate = None
if (training_mod["separate_mod"]):
    model_generate = modeler.Model(learning_rate=training_mod["learning_rate"]).to(device1)

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


def generator(threadID, queue: tmp.Queue, modelg, model_B, slow_control: tmp.Value, queue_B: tmp.Queue = None,
              model_C = None, queue_C: tmp.Queue = None):
    ''' A function for generator process
    :param threadID:
    :param queue:
    :param modelg:
    :param model_B: In double/half_doouble learning, model_B is trained.
    :param slow_control:
    '''

    if (training_mod["proportion_evaluation"]):
        if (training_mod["return_node"] == False):
            raise Exception(f"proportion_evaluation but not restart")
    if (training_mod["best_only"]):
        if (training_mod["return_node"] == False):
            raise Exception(f"best_only but not restart")

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
                child_node_lis_B = []
                root_node = None
                root_node_B = None
                random_B = False if random.random() > 0.5 else True
                random_another = random.random()
                ev = 0
                if (training_mod["play_method"] == "simple_play"):
                    dir, ev = playalg.simplePlay(state, modelg)
                elif (training_mod["play_method"] == "deep_play"):

                    if (training_mod["half_double_learning"]):
                        if not (training_mod["half_move"]):
                            dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                          greedy_value=training_mod["simple_value"],
                                                                          return_node=True,
                                                                          greedy_move=training_mod["greedy_move"],
                                                                          discount_factor=training_mod[
                                                                              "discount_factor"], double_learning=True,
                                                                          value_model=model_B)
                        else:
                            dir, ev, root_node = deep_play.expand_and_get(state, model_B, int(training_mod["depth"]),
                                                                          greedy_value=training_mod["simple_value"],
                                                                          return_node=True,
                                                                          greedy_move=training_mod["greedy_move"],
                                                                          discount_factor=training_mod[
                                                                              "discount_factor"], double_learning=True,
                                                                          value_model=modelg)

                    elif (training_mod["Double_learning"]):
                        if (training_mod["double_opposite"]):
                            dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                          greedy_value=training_mod["simple_value"],
                                                                          return_node=True,
                                                                          greedy_move=training_mod["greedy_move"],
                                                                          discount_factor=training_mod[
                                                                              "discount_factor"], double_learning=True,
                                                                          value_model=model_B)
                        else:
                            dir, ev, root_node = deep_play.expand_and_get(state, model_B, int(training_mod["depth"]),
                                                                          greedy_value=training_mod["simple_value"],
                                                                          return_node=True,
                                                                          greedy_move=training_mod["greedy_move"],
                                                                          discount_factor=training_mod[
                                                                              "discount_factor"], double_learning=True,
                                                                          value_model=modelg)
                    elif (training_mod["Original_double_learning"]):
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
                                                                                                            double_learning=True)
                    elif (training_mod["triple_learning"]):
                        dir, dir_A, dir_B, dir_C, ev_Abc, ev_Bca, ev_Cab, root_node = triple_deep_play.triple_expand_and_get(
                            state,
                            modelg,
                            model_B,
                            model_C,
                            int(training_mod["depth"]),
                            greedy_value=training_mod["simple_value"],
                            return_node=True,
                            greedy_move=training_mod["greedy_move"],
                            discount_factor=training_mod["discount_factor"],
                            triple_learning=True,
                        )
                    elif (training_mod["only_different"]):
                        dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                      greedy_value=training_mod["simple_value"],
                                                                      return_node=True,
                                                                      greedy_move=training_mod["greedy_move"],
                                                                      discount_factor=training_mod[
                                                                          "discount_factor"],
                                                                      device_number=device_number)
                        dir1, ev1, root_node1 = deep_play.expand_and_get(state, modelg, 1,
                                                                         greedy_value=training_mod["simple_value"],
                                                                         return_node=True,
                                                                         greedy_move=training_mod["greedy_move"],
                                                                         discount_factor=training_mod[
                                                                             "discount_factor"],
                                                                         device_number=device_number)
                    else:
                        if (training_mod["another_learning"] == False):
                            dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                          greedy_value=training_mod["simple_value"],
                                                                          return_node=True,
                                                                          greedy_move=training_mod["greedy_move"],
                                                                          discount_factor=training_mod[
                                                                              "discount_factor"],
                                                                          device_number=device_number)
                            if (training_mod["D3play_D1learn"]):
                                dir1, ev1, root_node1 = deep_play.expand_and_get(state, modelg, 1,
                                                                                 greedy_value=training_mod[
                                                                                     "simple_value"], return_node=True,
                                                                                 greedy_move=training_mod[
                                                                                     "greedy_move"],
                                                                                 discount_factor=training_mod[
                                                                                     "discount_factor"])
                        else:
                            if (random_another < training_mod["another_possibility"]):
                                # print("random_another")
                                if (training_mod["another_name"] == "D1"):
                                    dir, ev, root_node = deep_play.expand_and_get(state, modelg, 1,
                                                                                  greedy_value=training_mod[
                                                                                      "simple_value"], return_node=True,
                                                                                  greedy_move=training_mod[
                                                                                      "greedy_move"],
                                                                                  discount_factor=training_mod[
                                                                                      "discount_factor"])
                                elif (training_mod["another_name"] == "random"):
                                    dir, ev, root_node = deep_play.expand_and_get(state, modelg,
                                                                                  int(training_mod["depth"]),
                                                                                  greedy_value=training_mod[
                                                                                      "simple_value"], return_node=True,
                                                                                  greedy_move=training_mod[
                                                                                      "greedy_move"],
                                                                                  discount_factor=training_mod[
                                                                                      "discount_factor"])
                                    random_id = random.randint(0, len(root_node.children) - 1)
                                    best_child_node = root_node.children[random_id]
                                    root_node.next_node = best_child_node
                                    dir = root_node.available_actions[random_id]
                                    root_node.next_move = dir
                                    ev = best_child_node.exp + best_child_node.state.score - state.score
                                    root_node.exp = ev
                                else:
                                    raise Exception("Didn't find the random mod!")
                            else:
                                dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                              greedy_value=training_mod["simple_value"],
                                                                              return_node=True,
                                                                              greedy_move=training_mod["greedy_move"],
                                                                              discount_factor=training_mod[
                                                                                  "discount_factor"])
                    child_node_lis = root_node.children

                else:
                    # elif (training_mod["play_method"] == "renew_children"):
                    if (training_mod["half_double_learning"]):
                        dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                      return_node=True,
                                                                      greedy_move=training_mod["greedy_move"],
                                                                      random_state=training_mod["random_state"],
                                                                      discount_factor=training_mod["discount_factor"],
                                                                      double_learning=True, value_model=model_B)
                        child_node_lis = root_node.children
                    elif (training_mod["Double_learning"]):
                        dir, ev, root_node = deep_play.expand_and_get(state, model_B, int(training_mod["depth"]),
                                                                      return_node=True,
                                                                      greedy_move=training_mod["greedy_move"],
                                                                      random_state=training_mod["random_state"],
                                                                      discount_factor=training_mod["discount_factor"],
                                                                      double_learning=True, value_model=modelg)
                        child_node_lis = root_node.children
                    else:
                        dir, ev, root_node = deep_play.expand_and_get(state, modelg, int(training_mod["depth"]),
                                                                      return_node=True,
                                                                      greedy_move=training_mod["greedy_move"],
                                                                      random_state=training_mod["random_state"],
                                                                      discount_factor=training_mod["discount_factor"])
                        child_node_lis = root_node.children

                # Play and Log
                if (training_mod["actual_value"]):
                    if (training_mod["return_node"] == False):
                        raise Exception(f"Want actual_node but didnot return node")
                    after_node_lis.append(root_node)

                if (training_mod["return_node"] == True):
                    if (training_mod["testing_log"]):
                        if (training_mod["Double_learning"] or training_mod["half_double_learning"]):
                            testing_logger.info(
                                f" process_number {threadID} restart {restartcount} turn {turn} double_evaluation {root_node.double_exp} evaluation {root_node.exp} score {state.score} best_move {root_node.next_move} dir {dir} possible_move {root_node.available_actions}")
                        elif (training_mod["Original_double_learning"]):
                            testing_logger.info(
                                f" process_number {threadID} restart {restartcount} turn {turn} double_evaluation {root_node.exp_A_B} evaluationA_b{root_node.exp_Ab} evaluationB_a{root_node.exp_Ba} score {state.score} best_move {root_node.next_move_A_B} dir {dir} possible_move {root_node.available_actions}"
                            )
                        elif (training_mod["triple_learning"]):
                            testing_logger.info(
                                f" process_number {threadID} restart {restartcount} turn {turn} double_evaluation {root_node.exp_A_B_C} score {state.score} best_move {root_node.next_move_A_B_C} dir {dir} possible_move {root_node.available_actions}"
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

                if (training_mod["proportion_evaluation"] == True):
                    # do not support Double learning now.
                    child_EV_lis = []
                    child_V_lis = []
                    for child_node in child_node_lis:
                        child_EV_lis.append(child_node.exp)
                        child_V_lis.append(child_node.value)
                    if child_node_lis:
                        max_EV = max(child_EV_lis)
                        max_V = max(child_V_lis)
                        min_EV = min(child_EV_lis)
                        min_V = min(child_V_lis)
                        ave_EV = sum(child_EV_lis) / len(child_EV_lis)
                        ave_V = sum(child_V_lis) / len(child_V_lis)
                    if (training_mod["proportion_standardization"] == True):
                        std_EV = np.std(child_EV_lis)
                        std_V = np.std(child_V_lis)
                        scale = 1
                        if (std_EV != 0):
                            scale = std_V / std_EV

                if (training_mod["same_state"] == False):
                    state.play(dir)

                # Put data into queue
                if (training_mod["actual_value"] == False):
                    if (training_mod["play_method"] == "renew_children"):
                        if (training_mod["proportion_evaluation"] == True):

                            for child_node in child_node_lis:
                                if (training_mod["proportion_standardization"] == False):
                                    # target_value = (child_node.exp - ave_EV)* (ave_V/ave_EV) + ave_V
                                    target_value = (child_node.exp - ave_EV) + ave_V
                                else:

                                    target_value = (child_node.exp - ave_EV) * scale + ave_V
                                queue.put({'lastboard': child_node.state.board, 'target': target_value})
                        elif (training_mod["best_only"]):
                            if (training_mod["proportion_evaluation"] == True):
                                raise Exception(f"proportion_evaluation and best_only together")
                            queue.put({'lastboard': root_node.next_node.state.board, 'target': root_node.next_node.exp})

                        elif (training_mod["half_double_learning"] or training_mod["Double_learning"]):
                            for child_node in child_node_lis:
                                queue.put({'lastboard': child_node.state.board, 'target': child_node.double_exp})
                        else:
                            for child_node in child_node_lis:
                                queue.put({'lastboard': child_node.state.board, 'target': child_node.exp})
                    else:
                        # normal renew (D3)
                        if lastboard is not None:

                            if (training_mod["proportion_evaluation"] == True):
                                if (training_mod["proportion_standardization"] == False):

                                    # target_value = (root_node.next_node.exp - ave_EV) * (ave_V / ave_EV) + ave_V + \
                                    #                root_node.next_node.state.score - root_node.state.score
                                    target_value = (
                                                           root_node.next_node.exp - ave_EV) + ave_V + root_node.next_node.state.score - root_node.state.score
                                    queue.put({'lastboard': lastboard, 'target': target_value})
                                else:
                                    target_value = (
                                                           root_node.next_node.exp - ave_EV) * scale + ave_V + root_node.next_node.state.score - root_node.state.score
                                    queue.put({'lastboard': lastboard, 'target': target_value})
                            elif (training_mod["D3play_D1learn"]):
                                queue.put({'lastboard': lastboard, 'target': ev1})
                            elif (training_mod["Original_double_learning"]):
                                if (training_mod["renew_all"]):
                                    queue.put({'lastboard': lastboard, 'target': ev_Ba})
                                    queue_B.put({'lastboard': lastboard, 'target': ev_Ab})
                                else:
                                    random_number = random.random()
                                    if (random_number < 0.5):
                                        queue.put({'lastboard': lastboard, 'target': ev_Ba})
                                    else:
                                        queue_B.put({'lastboard': lastboard, 'target': ev_Ab})
                            elif (training_mod["triple_learning"]):

                                    random_number = 3*random.random()
                                    random_number_2 = random.random()
                                    if (random_number < 1):
                                        if(random_number_2 < 0.5):
                                            queue.put({'lastboard': lastboard, 'target': ev_Bca})
                                        else:
                                            queue.put({'lastboard': lastboard, 'target': ev_Cab})
                                    elif (random_number < 2):
                                        if (random_number_2 < 0.5):
                                            queue_B.put({'lastboard': lastboard, 'target': ev_Cab})
                                        else:
                                            queue_B.put({'lastboard': lastboard, 'target': ev_Abc})
                                    else:
                                        if (random_number_2 < 0.5):
                                            queue_C.put({'lastboard': lastboard, 'target': ev_Abc})
                                        else:
                                            queue_C.put({'lastboard': lastboard, 'target': ev_Bca})
                            elif (training_mod["only_different"]):
                                if(dir1 != dir):
                                    queue.put({'lastboard': lastboard, 'target': ev})
                            else:
                                queue.put({'lastboard': lastboard, 'target': ev})

                if not (training_mod["same_state"]):
                    lastboard = state.clone().board
                    state.putNewTile()
                    states.append(state.clone())
                else:
                    lastboard = root_node.next_node.state.board
                    state = root_node.next_node.children[0].state.clone()
                    states.append(state.clone())

                if state.isGameOver():
                    if (training_mod["log_name"] == "D1_nozero_fromzero"):
                        logger.info(
                            f'game over thread {threadID} restart {restartcount} score {state.score} length {len(states)}')
                        break

                    elif (training_mod["RC_end_AFstate"] or training_mod["play_method"] != "renew_children"):

                        if (training_mod["actual_value"] == False):
                            if (training_mod["Original_double_learning"]):
                                if (training_mod["renew_all"]):
                                    queue.put({'lastboard': lastboard, 'target': 0})
                                    queue_B.put({'lastboard': lastboard, 'target': 0})
                                else:
                                    random_number = random.random()
                                    if (random_number < 0.5):
                                        queue.put({'lastboard': lastboard, 'target': 0})
                                    else:
                                        queue_B.put({'lastboard': lastboard, 'target': 0})
                            elif (training_mod["triple_learning"]):

                                    random_number = 3*random.random()
                                    if (random_number < 1):
                                        queue.put({'lastboard': lastboard, 'target': 0})
                                    elif (random_number < 2):
                                        queue_B.put({'lastboard': lastboard, 'target': 0})
                                    else:
                                        queue_C.put({'lastboard': lastboard, 'target': 0})
                            else:
                                queue.put({'lastboard': lastboard, 'target': 0.0})

                    logger.info(
                        f'game over thread {threadID} restart {restartcount} score {state.score} length {len(states)}')
                    break

            if (training_mod["actual_value"]):
                if (after_node_lis == []):
                    raise Exception(f"after_node_lis is empty")
                after_node_lenth = len(after_node_lis)
                after_node_lis[-1].real_target = 0
                for nod_itr in range(2, after_node_lenth + 1):
                    reward = after_node_lis[-nod_itr + 1].state.score - after_node_lis[-nod_itr].state.score
                    after_node_lis[-nod_itr].real_target = after_node_lis[-nod_itr + 1].real_target + reward
                for node in after_node_lis:
                    queue.put({'lastboard': node.state.board, 'target': node.real_target})

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


def trainer(queue, modelt, model_B, slow_control: tmp.Value, training_device_number=None, name = ""):
    ''' A function for trainer thread
    :param queue:
    :param modelt:
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

        if (training_mod["half_double_learning"]):
            x = np.zeros([len(trainrecords), model_B.DIM_I], dtype="float32")
            y = torch.zeros([len(trainrecords), model_B.DIM_O], dtype=torch.float32).to(device1)
            for (i, playrecord) in enumerate(trainrecords):
                model_B.make_input(x[i, :], playrecord['lastboard'])
                y[i, :] = torch.tensor(playrecord['target']).to(device1)
            for batch in range(0, int(len(trainrecords) / cut_size) + 1):
                start_itr = batch * cut_size
                if (start_itr >= (len(trainrecords))):
                    break
                end_itr = (batch + 1) * cut_size
                if (end_itr >= (len(trainrecords))):
                    end_itr = len(trainrecords)
                model_B.train_mode(x_train=x[start_itr:end_itr], y_train=y[start_itr:end_itr], batch_size=cut_size)
            learned_actions += len(trainrecords)

            # logger.info(f'learned: {learned_actions:8,d} loss: {l:8,.2f}')
            logger.info(f'learned: {learned_actions:8,d}')

            # output parameters
            if ((learned_actions - len(trainrecords)) // OUTPUT_TIMING) < (learned_actions // OUTPUT_TIMING):
                # saver.save(sess, f'./{logdir}/weights', global_step=(learned_actions//OUTPUT_TIMING))
                if (training_mod["load_1000"]):
                    name_for_save = "./" + logdir + "/weights-" + str(learned_actions // OUTPUT_TIMING + 1000)

                    torch.save(model_B.state_dict(), name_for_save)
                    logger.info(f'saved parameters ./{logdir}/weights-{learned_actions // OUTPUT_TIMING + 1000}.*')
                else:
                    name_for_save = "./" + logdir + "/weights-" + str(learned_actions // OUTPUT_TIMING)
                    torch.save(model_B.state_dict(), name_for_save)
                    logger.info(f'saved parameters ./{logdir}/weights-{learned_actions // OUTPUT_TIMING}.*')
        elif (training_mod["step_learning"] or training_mod["Double_learning"]):
            x = np.zeros([len(trainrecords), model_B.DIM_I], dtype="float32")
            y = torch.zeros([len(trainrecords), model_B.DIM_O], dtype=torch.float32).to(device1)
            for (i, playrecord) in enumerate(trainrecords):
                model_B.make_input(x[i, :], playrecord['lastboard'])
                y[i, :] = torch.tensor(playrecord['target']).to(device1)

            for batch in range(0, int(len(trainrecords) / cut_size) + 1):
                start_itr = batch * cut_size
                if (start_itr >= (len(trainrecords))):
                    break
                end_itr = (batch + 1) * cut_size
                if (end_itr >= (len(trainrecords))):
                    end_itr = len(trainrecords)
                model_B.train_mode(x_train=x[start_itr:end_itr], y_train=y[start_itr:end_itr], batch_size=cut_size)
            learned_actions += len(trainrecords)

            # logger.info(f'learned: {learned_actions:8,d} loss: {l:8,.2f}')
            logger.info(f'learned: {learned_actions:8,d}')

            # output parameters
            if ((learned_actions - len(trainrecords)) // OUTPUT_TIMING) < (learned_actions // OUTPUT_TIMING):
                # saver.save(sess, f'./{logdir}/weights', global_step=(learned_actions//OUTPUT_TIMING))
                if (training_mod["load_1000"]):
                    name_for_save = "./" + logdir + "/weights-" + str(learned_actions // OUTPUT_TIMING + 1000)
                    torch.save(model_B.state_dict(), name_for_save)
                    logger.info(f'saved parameters ./{logdir}/weights-{learned_actions // OUTPUT_TIMING + 1000}.*')
                else:
                    name_for_save = "./" + logdir + "/weights-" + str(learned_actions // OUTPUT_TIMING)
                    torch.save(model_B.state_dict(), name_for_save)
                    logger.info(f'saved parameters ./{logdir}/weights-{learned_actions // OUTPUT_TIMING}.*')
        elif (training_mod["Original_double_learning"] or training_mod["triple_learning"]):
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
                modelt.train_mode(x_train=x[start_itr:end_itr], y_train=y[start_itr:end_itr], batch_size=cut_size,
                                  device_number=training_device_number)
            learned_actions += len(trainrecords)

            # logger.info(f'learned: {learned_actions:8,d} loss: {l:8,.2f}')
            logger.info(f'learned: {learned_actions:8,d}')

            # save parameters
            if ((learned_actions - len(trainrecords)) // OUTPUT_TIMING) < (learned_actions // OUTPUT_TIMING):
                if (training_mod["load_1000"]):
                    name_for_save = "./" + logdir + f"/weights-{name}-" + str(
                        learned_actions // OUTPUT_TIMING + 1000)
                    torch.save(modelt.state_dict(), name_for_save)
                    logger.info(
                        f'saved parameters ./{logdir}/weights-{training_device_number}-{learned_actions // OUTPUT_TIMING + 1000}.*')
                else:
                    name_for_save = "./" + logdir + f"/weights-{name}-" + str(
                        learned_actions // OUTPUT_TIMING)
                    torch.save(modelt.state_dict(), name_for_save)
                    logger.info(
                        f'saved parameters ./{logdir}/weights-{training_device_number}-{learned_actions // OUTPUT_TIMING}.*')
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
                if (training_mod["load_1000"]):
                    name_for_save = "./" + logdir + "/weights-" + str(learned_actions // OUTPUT_TIMING + 1000)
                    torch.save(modelt.state_dict(), name_for_save)
                    logger.info(f'saved parameters ./{logdir}/weights-{learned_actions // OUTPUT_TIMING + 1000}.*')
                else:
                    name_for_save = "./" + logdir + "/weights-" + str(learned_actions // OUTPUT_TIMING)
                    torch.save(modelt.state_dict(), name_for_save)
                    logger.info(f'saved parameters ./{logdir}/weights-{learned_actions // OUTPUT_TIMING}.*')

        if (training_mod["step_learning"]):
            current_step += 1
            if (current_step >= training_mod["step"]):
                current_step = 0
                modelt.load_state_dict(model_B.state_dict())


# generator_executor = concurrent.futures.ThreadPoolExecutor(max_workers=numprocess)
# for  i in range(numprocess): generator_executor.submit(generator(i))
# # generator(0)()
#
# print("start training")
# trainer()


if __name__ == '__main__':
    processes = []

    if (training_mod["triple_learning"]):
        numprocess = 2

    if (training_mod["load_1000"] == True):

        if (training_mod["double_mode"] == False):
            checkpointprefix = f'{logdir}/weights-1000'
            model.load_state_dict(torch.load(checkpointprefix, map_location=device0))
        else:
            checkpointprefix = f'{logdir}/weights-another'
            model.load_state_dict(torch.load(checkpointprefix, map_location=device0))

        if (training_mod["separate_mod"]):
            model_generate.load_state_dict(torch.load(checkpointprefix, map_location=device0))

        if (training_mod["Double_learning"] or training_mod["half_double_learning"] or training_mod["step_learning"]):
            if (training_mod["different_weight"]):
                checkpointprefix = f'{logdir}/weights-another'
            model_B.load_state_dict(torch.load(checkpointprefix, map_location=device0))
        elif (training_mod["Original_double_learning"]):
            if (training_mod["different_weight"]):
                checkpointprefix = f'{logdir}/weights-another'
            model_B.load_state_dict(torch.load(checkpointprefix, map_location=device1))
        elif (training_mod["triple_learning"]):
            if (training_mod["different_weight"]):
                checkpointprefix = f'{logdir}/weights-another'
            model_B.load_state_dict(torch.load(checkpointprefix, map_location=device1))
            if (training_mod["different_weight"]):
                checkpointprefix = f'{logdir}/weights-another2'
                model_C.load_state_dict(torch.load(checkpointprefix, map_location=device1))

    if (training_mod["freeze"]):
        modeler.freeze(model)
        if (training_mod["Double_learning"] or training_mod["half_double_learning"] or training_mod["step_learning"] or
                training_mod["Original_double_learning"]):
            modeler.freeze(model_B)

    model.share_memory()
    if (training_mod["Double_learning"] or training_mod["half_double_learning"] or training_mod["step_learning"] or
            training_mod["Original_double_learning"]):
        model_B.share_memory()
    if(training_mod["triple_learning"]):
        model_B.share_memory()
        model_C.share_memory()
    slow_control = tmp.Value('d', 0.0)
    # now, slow control is useless

    for i in range(numprocess):
        if (training_mod["separate_mod"]):
            processes.append(
                tmp.Process(target=generator, args=(i, queue1, model_generate, model_B, slow_control,)))
        elif (training_mod["Original_double_learning"]):
            processes.append(
                tmp.Process(target=generator, args=(i, queue1, model, model_B, slow_control, queue2)))
        elif (training_mod["triple_learning"]):
            processes.append(
                tmp.Process(target=generator, args=(i, queue1, model, model_B, slow_control, queue2, model_C, queue3)))
        else:
            processes.append(tmp.Process(target=generator, args=(i, queue1, model, model_B, slow_control,)))
    for process in processes:
        process.start()

    if (training_mod["Original_double_learning"]):
        training_process1 = tmp.Process(target=trainer, args=(queue1, model, None, slow_control, device_number))
        training_process2 = tmp.Process(target=trainer,
                                        args=(queue2, model_B, None, slow_control, another_device_number))
        training_process1.start()
        training_process2.start()
    elif (training_mod["triple_learning"]):
        training_process1 = tmp.Process(target=trainer, args=(queue1, model, None, slow_control, device_number, "A"))
        training_process2 = tmp.Process(target=trainer,
                                        args=(queue2, model_B, None, slow_control, another_device_number, "B"))
        training_process3 = tmp.Process(target=trainer,
                                        args=(queue3, model_C, None, slow_control, another_device_number, "C"))
        training_process1.start()
        training_process2.start()
        training_process3.start()
    else:
        training_process = tmp.Process(target=trainer, args=(queue1, model, model_B, slow_control, device_number))
        training_process.start()

    for process in processes:
        process.join()

    if (training_mod["Original_double_learning"]):
        training_process1.join()
        training_process2.join()
    elif(training_mod["triple_learning"]):
        training_process1.join()
        training_process2.join()
        training_process3.join()
    else:
        training_process.join()
