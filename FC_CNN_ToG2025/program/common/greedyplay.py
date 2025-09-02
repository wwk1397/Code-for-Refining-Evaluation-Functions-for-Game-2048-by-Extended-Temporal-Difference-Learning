import os, sys
# greedy play with gpu0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import numpy as np
import logging
import threading, concurrent.futures
import queue

import random

sys.path.append('../common')
import Game2048
device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# usage: python greedyplay.py 100 cnn22B ??/weights-40 50
# usage: python greedyplay.py 100 cnn22B ../exp/training_cnn22B_1/weights-1 50
# sys.argv = "greedyplay.py 100 cnn22B ../exp/training_cnn22B_1/weights_1 50".split(" ")

seed = int(sys.argv[1])
modelname = sys.argv[2]
checkpointprefix = sys.argv[3]
NUM_GAMES = int(sys.argv[4])

num_thread = 5

base = modelname
# exec(f'import {base} as modeler')

import cnn22B as modeler
model = modeler.Model().to(device0)

logfile = f'{checkpointprefix}-greedy-s{seed}.log'

#乱数
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
np.set_printoptions(threshold=np.inf)

# if not os.path.exists(logdir): os.makedirs(logdir)

# ログ関連
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(logfile)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s'))
logger.addHandler(fh)
logger.info('start')

# セッション用意
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver(max_to_keep=None)
# saver.restore(sess, checkpointprefix)

def map_location(storage, _):
    return storage.cuda(device0.index if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load(checkpointprefix, map_location=map_location))

logger.info(f'model restored')

queueA = queue.Queue(NUM_GAMES)

import playalg
# ゲームプレイスレッド
def gameplay(gameID):
    def gameplayCore():
        # 1 ゲーム分プレイする
        state = Game2048.State()
        state.initGame()
        turn = 0
        while True:
            turn += 1
            dir,ev = playalg.simplePlay(state, model)
            state.play(dir)
            state.putNewTile()
            if state.isGameOver():
                logger.info(f'game over {gameID} score {state.score} turn {turn} maxtile {max(state.board)}')
                queueA.put({'id':gameID, 'score':state.score, 'turn':turn, 'maxtile':max(state.board)})
                break
    return gameplayCore

with concurrent.futures.ThreadPoolExecutor(max_workers=num_thread) as player_executor:
    for  i in range(NUM_GAMES): player_executor.submit(gameplay(i))
# gameplay(0)()

results = []
for i in range(NUM_GAMES): results.append(queueA.get())

logger.info(f'After {NUM_GAMES} games')
logger.info(f'Average score: {np.mean([r["score"] for r in results])}')
logger.info(f'Max score: {max([r["score"] for r in results])}')
logger.info(f'Min score: {min([r["score"] for r in results])}')
logger.info(f'32768 clear count: {sum([1 if r["maxtile"] >= 15 else 0 for r in results])}')
logger.info(f'16384 clear count: {sum([1 if r["maxtile"] >= 14 else 0 for r in results])}')
logger.info(f' 8192 clear count: {sum([1 if r["maxtile"] >= 13 else 0 for r in results])}')
logger.info(f' 4096 clear count: {sum([1 if r["maxtile"] >= 12 else 0 for r in results])}')
logger.info(f' 2048 clear count: {sum([1 if r["maxtile"] >= 11 else 0 for r in results])}')

