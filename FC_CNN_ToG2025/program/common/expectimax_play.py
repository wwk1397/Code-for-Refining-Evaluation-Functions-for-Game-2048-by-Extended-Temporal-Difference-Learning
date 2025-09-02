# -*- coding: utf-8 -*-


import sys, os, random, logging, torch
# sys.argv = "expectimax_play.py 100 D2 0 0 1 1000 20 RCS".split(" ")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpu=int(sys.argv[3])
device_number = 0
if gpu>=0:
    device_number = gpu

if gpu==-1:
    device_number = -1
    device = torch.device(f'cpu')
else:
    device = torch.device(f'cuda:{device_number}')
import numpy as np

import torch
import time
from datetime import datetime
#sys.path.append('../common')
import Game2048

mod_name = [
    "",
    "AE-CNN",
    "BP-CNN",
    "CNN",
    "CNN-CNN",
]

# usage: python testing_state.py (seed) (nn_calls)
# usage: python testing_state.py 1 N100
# usage: python testing_state.py 1 D3

# usage: python  expectimax_play.py   1      D3    0    0-3      1-4       250       30
#        python  expectimax_play.py   seed  D/N? gpu modname thread_num   weight_num
#nohup python expectimax_play.py 100 D2 0 0 1 1 20 > out.out &


episodes_num = int(sys.argv[7])
seed = int(sys.argv[1])
if sys.argv[2][0] == 'N':
    nn_calls = int(sys.argv[2][1:])
    maxdepth = 0
else:
    nn_calls = 0
    maxdepth = int(sys.argv[2][1:])



# checkpointprefix = "../"+mod_name [int(sys.argv[4])] + "-" + sys.argv[5] +\
#                    "/program/exp/training_cnn22B_"+ sys.argv[5] + "/weights-" + sys.argv[6]

# checkpointprefix = f"../exp/training{sys.argv[5]}_cnn22B_"+ sys.argv[5] + "/weights-" + sys.argv[6]
# training_name = [
# "deep_move_greedy_value_from1000",
# "deep_move_3_from1000",
# "renew_children_separate",
# ]
checkpointprefix_A = ""
checkpointprefix_B = ""
if((sys.argv[8]).__contains__("double")):
    checkpointprefix = f"../exp/{(sys.argv[8])}/training_cnn22B_" + sys.argv[5] + "/weights-A-" + sys.argv[6]
    checkpointprefix_A = f"../exp/{(sys.argv[8])}/training_cnn22B_"+ sys.argv[5] + "/weights-A-" + sys.argv[6]
    checkpointprefix_B = f"../exp/{(sys.argv[8])}/training_cnn22B_" + sys.argv[5] + "/weights-B-" + sys.argv[6]
else:
    checkpointprefix = f"../exp/{(sys.argv[8])}/training_cnn22B_" + sys.argv[5] + "/weights-" + sys.argv[6]

expectimaxdepth = maxdepth

logfile = f'{checkpointprefix}-expectimax{expectimaxdepth}-s{seed}.log'

import re
if os.path.exists(logfile):
    with open(logfile, 'r') as log:
        log_content = log.read()
        game_over_count = len(re.findall(r"GameOver", log_content))
        if game_over_count > 0:
            episodes_num -= game_over_count
            seed += 1
            sys.argv[1] = str(seed)
            print(f"Found {game_over_count} completed games in log. Adjusted episodes_num to {episodes_num}, seed to {seed}.")


# expectimaxdepth = int(sys.argv[7])

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(logfile)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger.addHandler(fh)

logger.info(f'Execution parameters: {sys.argv}')

# Preparing seed, model and session
# random.seed(seed)
# np.random.seed(seed)
# tf.set_random_seed(seed)
# np.set_printoptions(threshold=np.inf)

random.seed(seed)
np.random.seed(seed)
# tc.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

from cnn22B import Model,policy_value,restore_mod


# restore_mod(checkpointprefix=checkpointprefix)

if((sys.argv[8]).__contains__("double")):
    policy_value_net_A = Model().to(device)
    policy_value_net_B = Model().to(device)
    policy_value_net_A.load_state_dict(torch.load(checkpointprefix_A, map_location=device))
    policy_value_net_B.load_state_dict(torch.load(checkpointprefix_B, map_location=device))
else:
    policy_value_net = Model().to(device)
    policy_value_net.load_state_dict(torch.load(checkpointprefix, map_location=device))


from expectimax import *
Expectimax_setting.nn_calls = nn_calls
Expectimax_setting.maxdepth = maxdepth
Expectimax_setting.policy_value = policy_value

import deep_play, double_deep_play

def run():
    for i in range(episodes_num):
        player=expectimax()
        state = Game2048.State()
        state.initGame()
        # player.update_with_move(-1)

        while True:
            # move = player.get_move(state.clone())
            if ((sys.argv[8]).__contains__("double")):

                # move, _, _, _, _, _ = double_deep_play.double_expand_and_get(state, policy_value_net_A, policy_value_net_B,int(expectimaxdepth) * 2 - 1,
                #                                                              greedy_value=False,
                #                                                              device_0_number=device_number,
                #                                                              device_1_number=device_number,
                #                                                              # batch_size=96,
                #                                                             )
                move, ev = deep_play.expand_and_get(state, policy_value_net_A, int(expectimaxdepth) * 2 - 1,
                                                    greedy_value=False, device_number=device_number, quick=False,
                                                    batch_size=96)



            else:
                move, ev = deep_play.expand_and_get(state, policy_value_net, int(expectimaxdepth) * 2 - 1,
                                                    greedy_value=False, device_number=device_number, quick=False, batch_size=96)

            # perform a move
#            logger.info(f'board = {state.board}, score = {state.score}')
#            logger.info(f'select={"NESW"[move]}')
#            logger.info(f'model states evaluated = {Expectimax_setting.model_call_states}')
#            logger.info(f'maximum simulation depth = {Expectimax_setting.depth}')
#            logger.info(str(sorted([(key, int(n.exp)) for (key, (n, r)) in player._root._children.items()])))
            #Expectimax_setting.reset_status()

            # state.print()
#            print(f'select={"NESW"[move]}')
#            print()

            state.play(move)
            state.putNewTile()
            # player.update_with_move(-1)

            # self.board.do_move(move)
            if state.isGameOver():
                logger.info(f'GameOver {state.score} {state.board}')
                break

if __name__=='__main__':
    run()
    # print(checkpointprefix)
