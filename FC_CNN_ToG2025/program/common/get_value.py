#get value+e from expectimax in training.
import numpy as np
import cnn22B as modeler
from Game2048 import State
from typing import List, Dict

INPUT_SIZE = modeler.INPUT_SIZE



class Expand_state():

    def __init__(self, state:State, depth:int,):

        self.root_state = state
        self.type = "state"
        self.input_lenth = 32**(int(depth)/2)*(4**(int(depth)/2 + int(depth)%2))

        self.input_list = np.zeros((self.input_lenth, INPUT_SIZE), dtype='float32')
        self.reward_list = np.zeros((self.input_lenth, INPUT_SIZE), dtype='float32')
        self.possibility_list = np.ones((self.input_lenth, 1), dtype="float32")
        self.belong = np.ones((self.input_lenth, 1), dtype="int")
        self.value_list = np.zeros((self.input_lenth,), dtype="float32")
        # 0,1,2,3 -> up, right, down, left
        self.start_idx = 0

        self.end_idx = self.start_idx + self.input_lenth - 1
        self.expanded = False

    def make_input(self,x,board):
        # encode board into x
        for j in range(16):
            b = board[j]
            x[16 * b + j] = 1
            x[16 * (18 + int(j//4) ) + j ] = 1
            x[16 * (22 + j%4) + j  ] = 1

    def expand(self, start_idx:int, depth:int, lenth:int, current_possibility: float, current_state: State,
               current_belong: int = -1,):
        if depth == 0:
            # final depth
            if lenth != 1:
                raise Exception("In expanding part, endstate lenth != 1")
            self.make_input(self.input_list[start_idx,:], current_state.board)
            self.belong[start_idx] = current_belong
            self.possibility_list[start_idx] = current_possibility
            self.reward_list[start_idx] = current_state.score - self.root_state.score
            return

        if depth %2 == 0:
            # state, but not in final depth
            after_state_lis = [
                current_state.clone().doUp(),
                current_state.clone().doRight(),
                current_state.clone().doDown(),
                current_state.clone().doLeft(),
            ]
            # 0,1,2,3 -> up, right, down, left

            for direction in range(4):
                if(current_belong == -1):
                    next_belong = direction
                else:
                    next_belong = current_belong
                if(lenth%4 != 0):
                    raise Exception("In expanding part, lenth%4 != 0")
                next_lenth = int(lenth/4)

                self.expand(
                    start_idx=start_idx + direction * next_lenth,
                    depth= depth-1,
                    lenth= next_lenth,
                    current_possibility= current_possibility/4,
                    current_state= after_state_lis[direction],
                    current_belong= next_belong,
                )

        if depth %2 == 1:
            # after state, but not in final depth
            return



        self.expanded = True


                            
        
