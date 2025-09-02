# get value+e from expectimax in training.
import numpy as np
import torch

import cnn22B as modeler
from Game2048 import State
from typing import List, Dict
import copy
import random

INPUT_SIZE_A = modeler.INPUT_SIZE
INPUT_SIZE_B = modeler.INPUT_SIZE
INPUT_SIZE_C = modeler.INPUT_SIZE


class triple_exp_node(object):
    def __init__(self, state: State, height: int, type="state", possiblity=1.0):
        self.children = []
        self.type = type
        self.exp_A = -float("Inf")
        self.exp_B = -float("Inf")
        self.exp_C = -float("Inf")
        self.exp_A_B_C = -float("Inf")
        # exp_A_B_C means (exp_A + exp_B + exp_C)/3. Only been calculated from root node.
        self.exp_Abc = -float("Inf")
        # a, b, and c means a* b*, and c*
        self.exp_Bca = -float("Inf")
        self.exp_Cab = -float("Inf")
        self.value_A = -float("Inf")
        self.value_B = -float("Inf")
        self.value_C = -float("Inf")
        self.height = height
        self.state = state
        self.expanded = False
        self.possibility = possiblity
        self.game_over = False
        self.available_actions = []
        self.next_move_A = -1
        self.next_move_B = -1
        self.next_move_C = -1
        self.next_move_A_B_C = -1
        # best move from exp_A_B. Only been calculated from root node
        self.next_node_A = None
        self.next_node_B = None
        self.next_node_C = None
        self.next_node_A_B_C = None
        self.real_target = -float("Inf")



def triple_expand(current_node: triple_exp_node, bottom_nodes: List[triple_exp_node], random_state=False) -> List[triple_exp_node]:
    current_board_state = current_node.state
    # print("**", current_node.state is None)
    if (current_node.state.isGameOver()):
        current_node.game_over = True
        current_node.value_A = 0
        current_node.value_B = 0
        current_node.value_C = 0
    elif (current_node.height == 0):
        bottom_nodes.append(current_node)
        current_node.game_over = True

    elif current_node.type == "state":
        # "state", but not the end_nodes

        for direction in range(4):
            if current_board_state.canMoveTo(direction):
                child_board_state = current_board_state.clone()
                child_board_state.play(direction)
                current_node.available_actions.append(direction)
                child_node = triple_exp_node(state=child_board_state, height=current_node.height - 1,
                                             type="after_state")
                current_node.children.append(child_node)

    else:
        # "after_state", but not the end_nodes
        if not random_state:
            for position in range(16):
                if current_node.state.board[position] != 0:
                    continue
                add_triple_child_node(1, position, 0.9, current_node)
                add_triple_child_node(2, position, 0.1, current_node)
        else:
            # "random after->state"
            children_position_lis = []
            for position in range(16):
                if current_node.state.board[position] != 0:
                    continue
                children_position_lis.append(position)
            children_position_len = len(children_position_lis)

            position_random = random.randint(0, children_position_len-1)
            child_position = children_position_lis[position_random]
            tile_random = random.random()
            if(tile_random >= 0.9):
                add_triple_child_node(2, child_position, 1, current_node)
            else:
                add_triple_child_node(1, child_position, 1, current_node)

    return current_node.children


def add_triple_child_node(tile_value: int, position: int, possibility: float, node: triple_exp_node):
    child_board_state = node.state.clone()
    child_board_state.board[position] = tile_value
    child_node = triple_exp_node(state=child_board_state, height=node.height - 1,
                                 type="state", possiblity=possibility)
    node.children.append(child_node)


def calculate_triple_value(board: np.ndarray, model_A, model_B, model_C) -> (np.ndarray, np.ndarray, np.ndarray):
    # get value for board
    # board : np.array (seq_len,16)
    board_lenth = board.shape[0]
    input_boards_A = np.zeros((board_lenth, INPUT_SIZE_A), dtype="float32")
    input_boards_B = np.zeros((board_lenth, INPUT_SIZE_B), dtype="float32")
    input_boards_C = np.zeros((board_lenth, INPUT_SIZE_C), dtype="float32")

    for itr in range(board_lenth):
        model_A.make_input(input_boards_A[itr, :], board[itr])
        model_B.make_input(input_boards_B[itr, :], board[itr])
        model_C.make_input(input_boards_C[itr, :], board[itr])
    answer_A = model_A.predict(input_boards_A, 0).cpu().detach()
    answer_B = model_B.predict(input_boards_B, 1).cpu().detach()
    answer_C = model_C.predict(input_boards_C, 1).cpu().detach()

    return answer_A, answer_B, answer_C


def get_triple_node_exp_first(
        current_node: triple_exp_node,
        discount_factor=1,
        random_state=False,
        greedy_move=False
) -> (float, float, float):
    # get exp_A and exp_B
    # we will get exp_Ab, exp_Ba from get_truple_node_exp_second
    if (current_node.game_over == True):
        if (current_node.value_B == -float("inf") or current_node.value_A == -float("inf") or current_node.value_C == -float("inf")):
            raise Exception(f"One end node value is not calculated at height {current_node.height}")
        current_node.exp_A = discount_factor * current_node.value_A
        current_node.exp_B = discount_factor * current_node.value_B
        current_node.exp_C = discount_factor * current_node.value_C
        return current_node.exp_A, current_node.exp_B, current_node.exp_C

    if (current_node.type == "after_state"):
        children_exp_reward_lis_A = []
        children_exp_reward_lis_B = []
        children_exp_reward_lis_C = []
        for child_node in current_node.children:
            exp_A ,exp_B, exp_C = get_triple_node_exp_first(child_node, discount_factor, random_state, greedy_move)
            children_exp_reward_lis_A.append(child_node.possibility * exp_A)
            children_exp_reward_lis_B.append(child_node.possibility * exp_B)
            children_exp_reward_lis_C.append(child_node.possibility * exp_C)
        if not random_state:
            current_node.exp_A = 2 * sum(children_exp_reward_lis_A) / len(children_exp_reward_lis_A)
            current_node.exp_B = 2 * sum(children_exp_reward_lis_B) / len(children_exp_reward_lis_B)
            current_node.exp_C = 2 * sum(children_exp_reward_lis_C) / len(children_exp_reward_lis_C)
        else:
            current_node.exp_A = sum(children_exp_reward_lis_A) / len(children_exp_reward_lis_A)
            current_node.exp_B = sum(children_exp_reward_lis_B) / len(children_exp_reward_lis_B)
            current_node.exp_C = sum(children_exp_reward_lis_C) / len(children_exp_reward_lis_C)
        return current_node.exp_A, current_node.exp_B, current_node.exp_C

    if (current_node.type == "state"):

        best_act_A = -1
        best_act_B = -1
        best_act_C = -1
        best_act_ABC = -1
        best_node_A = None
        best_node_B = None
        best_node_C = None
        best_node_ABC = None
        best_value_A = -float("inf")
        best_value_B = -float("inf")
        best_value_C = -float("inf")
        best_value_ABC = -float("inf")
        for itr, child_node in enumerate(current_node.children):
            child_exp_A, child_exp_B, child_exp_C = get_triple_node_exp_first(child_node, discount_factor, random_state, greedy_move)
            children_exp_reward_A = child_exp_A + child_node.state.score - current_node.state.score
            children_exp_reward_B = child_exp_B + child_node.state.score - current_node.state.score
            children_exp_reward_C = child_exp_C + child_node.state.score - current_node.state.score
            children_exp_reward_ABC = (children_exp_reward_A + children_exp_reward_B + children_exp_reward_C) / 3

            if (greedy_move):
                if (child_node.value == -float("Inf")):
                    raise Exception("simple move but deeper than 3")
                children_exp_reward_A = child_node.state.score - current_node.state.score + child_node.value_A
                children_exp_reward_B = child_node.state.score - current_node.state.score + child_node.value_B
                children_exp_reward_C = child_node.state.score - current_node.state.score + child_node.value_C
                children_exp_reward_ABC = (children_exp_reward_A + children_exp_reward_B + children_exp_reward_C) / 3

            if (children_exp_reward_A > best_value_A):
                best_act_A = current_node.available_actions[itr]
                best_value_A = children_exp_reward_A
                best_node_A = child_node
            if (children_exp_reward_B > best_value_B):
                best_act_B = current_node.available_actions[itr]
                best_value_B = children_exp_reward_B
                best_node_B = child_node
            if (children_exp_reward_C > best_value_C):
                best_act_C = current_node.available_actions[itr]
                best_value_C = children_exp_reward_C
                best_node_C = child_node
            if (children_exp_reward_ABC > best_value_ABC):
                best_act_ABC = current_node.available_actions[itr]
                best_value_ABC = children_exp_reward_ABC
                best_node_ABC = child_node
            # print(children_exp_reward, ",", best_act,",",best_value)

        current_node.next_move_A = best_act_A
        current_node.next_move_B = best_act_B
        current_node.next_move_C = best_act_C
        current_node.next_move_A_B_C = best_act_ABC
        current_node.next_node_A = best_node_A
        current_node.next_node_B = best_node_B
        current_node.next_node_C = best_node_C
        current_node.next_node_A_B_C = best_node_ABC
        # print("best_act:",best_act)
        current_node.exp_A = best_value_A
        current_node.exp_B = best_value_B
        current_node.exp_C = best_value_C
        current_node.exp_A_B_C = best_value_ABC
        return current_node.exp_A, current_node.exp_B, current_node.exp_C


def get_triple_node_exp_second(
        current_node: triple_exp_node,
        discount_factor=1,
        random_state=False,
        greedy_move=False
) -> (float, float, float):
    # get exp_Abc and exp_Bca exp Cab
    # we have got next_move_A,next_move_B,next_node_A,next_node_B,exp_A,exp_B, ... for all nodes with get_triple_node_exp_first
    if (current_node.game_over == True):
        if (current_node.value_B == -float("inf") or current_node.value_A == -float("inf") or current_node.value_C == -float("inf")):
            raise Exception(f"One end node value is not calculated at height {current_node.height}")
        current_node.exp_Abc = discount_factor * current_node.value_A
        current_node.exp_Bca = discount_factor * current_node.value_B
        current_node.exp_Cab = discount_factor * current_node.value_C
        return current_node.exp_Abc, current_node.exp_Bca, current_node.exp_Cab

    if (current_node.type == "after_state"):
        children_exp_reward_lis_Abc = []
        children_exp_reward_lis_Bca = []
        children_exp_reward_lis_Cab = []
        for child_node in current_node.children:
            exp_Abc ,exp_Bca, exp_Cab = get_triple_node_exp_second(child_node, discount_factor, random_state, greedy_move)
            children_exp_reward_lis_Abc.append(child_node.possibility * exp_Abc)
            children_exp_reward_lis_Bca.append(child_node.possibility * exp_Bca)
            children_exp_reward_lis_Cab.append(child_node.possibility * exp_Cab)
        if not random_state:
            current_node.exp_Abc = 2 * sum(children_exp_reward_lis_Abc) / len(children_exp_reward_lis_Abc)
            current_node.exp_Bca = 2 * sum(children_exp_reward_lis_Bca) / len(children_exp_reward_lis_Bca)
            current_node.exp_Cab = 2 * sum(children_exp_reward_lis_Cab) / len(children_exp_reward_lis_Cab)
        else:
            current_node.exp_Abc = sum(children_exp_reward_lis_Abc) / len(children_exp_reward_lis_Abc)
            current_node.exp_Bca = sum(children_exp_reward_lis_Bca) / len(children_exp_reward_lis_Bca)
            current_node.exp_Cab = sum(children_exp_reward_lis_Cab) / len(children_exp_reward_lis_Cab)
        return current_node.exp_Abc, current_node.exp_Bca, current_node.exp_Cab

    if (current_node.type == "state"):

        best_next_node_A = current_node.next_node_A
        best_next_node_B = current_node.next_node_B
        best_next_node_C = current_node.next_node_C

        if(current_node.height %2 == 0):
            raise Exception(f"State but height:{current_node.height}")

        if(current_node.height == 1):
            exp_Abc, _, _ = get_triple_node_exp_second(best_next_node_C, discount_factor, random_state, greedy_move)
            _, exp_Bca, _ = get_triple_node_exp_second(best_next_node_A, discount_factor, random_state, greedy_move)
            _, _, exp_Cab = get_triple_node_exp_second(best_next_node_B, discount_factor, random_state, greedy_move)

            current_node.exp_Abc = exp_Abc + best_next_node_C.state.score - current_node.state.score
            current_node.exp_Bca = exp_Bca + best_next_node_A.state.score - current_node.state.score
            current_node.exp_Cab = exp_Cab + best_next_node_B.state.score - current_node.state.score

        if(current_node.height == 3):
            exp_Abc, _, _ = get_triple_node_exp_second(best_next_node_B, discount_factor, random_state, greedy_move)
            _, exp_Bca, _ = get_triple_node_exp_second(best_next_node_C, discount_factor, random_state, greedy_move)
            _, _, exp_Cab = get_triple_node_exp_second(best_next_node_A, discount_factor, random_state, greedy_move)

            current_node.exp_Abc = exp_Abc + best_next_node_B.state.score - current_node.state.score
            current_node.exp_Bca = exp_Bca + best_next_node_C.state.score - current_node.state.score
            current_node.exp_Cab = exp_Cab + best_next_node_A.state.score - current_node.state.score

        return current_node.exp_Abc, current_node.exp_Bca, current_node.exp_Cab


def triple_expand_and_get(root_state: State, model_A, model_B, model_C, height=3, greedy_value=False, return_node=False,
                          greedy_move=False, random_state= False, discount_factor=1, triple_learning=False,
                          device_number=None,
                          ):
    """
    :param device_number:
    :param root_state:
    :param model:
    :param height:
    :param greedy_value:
    :param return_node:
    :param greedy_move:
    :param random_state:
    :param discount_factor:
    :param triple_learning: Use triple learning or not. If so, we use "model" to select move a*, use value model to get
        value from move a*.
    :param value_model:  Only used in triple_learning.
    :return:
    """
    if (model_B == None or model_A == None or model_C == None):
        raise Exception(f"It is not the original triple learning")

    root_node = triple_exp_node(state=root_state.clone(), height=height)
    standby_node_lis = [root_node]
    bottom_node_lis = []

    # expand
    while (standby_node_lis != []):
        children_node_lis = []
        for node in standby_node_lis:
            children_node_lis.extend(triple_expand(node, bottom_nodes=bottom_node_lis, random_state=random_state))
        standby_node_lis = children_node_lis

    bottom_board_lis = []
    # calculate bottom_value
    for bottom_node in bottom_node_lis:
        bottom_board_lis.append(bottom_node.state.board)

    if greedy_value or return_node or greedy_move:
        # calculate Value(children)
        for cildren_node in root_node.children:
            bottom_node_lis.append(cildren_node)
            bottom_board_lis.append(cildren_node.state.board)

    bottom_board_array = np.array(bottom_board_lis)
    bottom_answer_array = calculate_triple_value(bottom_board_array, model_A, model_B, model_C)
    for itr, bottom_node in enumerate(bottom_node_lis):
        bottom_node.value_A = bottom_answer_array[0][itr].item()
        bottom_node.value_B = bottom_answer_array[1][itr].item()
        bottom_node.value_C = bottom_answer_array[2][itr].item()

    # print("***:",len(root_node.children) )
    ev_A, ev_B, ev_C = get_triple_node_exp_first(current_node=root_node, discount_factor=discount_factor,random_state=random_state, greedy_move=greedy_move)
    ev_Abc, ev_Bca, ev_Cab = get_triple_node_exp_second(current_node=root_node, discount_factor=discount_factor, random_state=random_state,
                                                greedy_move=greedy_move)
    #We should add more in greedy move
    return root_node.next_move_A_B_C, root_node.next_move_A, root_node.next_move_B, root_node.next_move_C, ev_Abc, ev_Bca, ev_Cab, root_node




if __name__ == '__main__':
    board = np.ones((2, 16), dtype="int")
    input_boards = np.zeros((board.shape[0], INPUT_SIZE_A), dtype="float32")
    # for itr in range(2):
    # make_input(input_boards[itr,:],board[itr,:])
    print(input_boards)
