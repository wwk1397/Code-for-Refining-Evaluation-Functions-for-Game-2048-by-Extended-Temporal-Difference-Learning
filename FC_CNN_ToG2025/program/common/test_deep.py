import sys, os, random, logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

import torch
import time
from datetime import datetime
#sys.path.append('../common')
import Game2048
import deep_play
import cnn22B
logdir = f'test_deep_play'
if not os.path.exists(logdir): os.makedirs(logdir)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
np.set_printoptions(threshold=np.inf)

# logfile = f'{logdir}/deep_play.log'
# logfile = f'{logdir}/simple_play.log'
logfile = f'{logdir}/deep3_play.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(logfile)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger.addHandler(fh)

logger.info(f'Execution parameters: {sys.argv}')

game_number = 1

def run():
    model = cnn22B.Model().cuda()
    checkpointprefix = f'weights-1000'
    model.load_state_dict(torch.load(checkpointprefix))

    for i in range(game_number):
        state = Game2048.State()
        state.initGame()
        after_node_lis = []
        while True:
            # move = player.get_move(state.clone())
            # move,ev = deep_play.expand_and_get(state.clone(),model,3)
            move, ev, root_node = deep_play.expand_and_get(state.clone(), model, 1, return_node=True)

            state.play(move)
            state.putNewTile()

            if state.isGameOver():
                logger.info(f'GameOver {state.score} {state.board}')
                break

if __name__ == '__main__':
    run()