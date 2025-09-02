# get value+e from expectimax in training.
import numpy as np
import torch

import cnn22B as modeler
from Game2048 import State
from typing import List, Dict
import copy
import random, math

INPUT_SIZE = modeler.INPUT_SIZE


class exp_node(object):
    def __init__(self, state: State, height: int, type="state", possiblity=1.0):
        self.children = []
        self.type = type
        self.exp = -float("Inf")
        self.value = -float("Inf")
        self.double_value = -float("Inf")
        self.double_exp = -float("Inf")
        self.height = height
        self.state = state
        self.expanded = False
        self.possibility = possiblity
        self.game_over = False
        self.available_actions = []
        self.next_move = -1
        self.next_node = None
        self.real_target = -float("Inf")



def expand(current_node: exp_node, bottom_nodes: List[exp_node], random_state=False, state_percentage = 1.0) -> List[exp_node]:
    current_board_state = current_node.state
    # print("**", current_node.state is None)
    current_node.children = []


    if (current_node.state.isGameOver()):
        current_node.game_over = True
        current_node.value = 0
        # double value is used in double learning
        current_node.double_value = 0

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
                child_node = exp_node(state=child_board_state, height=current_node.height - 1,
                                      type="after_state")
                current_node.children.append(child_node)

    else:
        # "after_state", but not the end_nodes
        if not random_state:
            for position in range(16):
                if current_node.state.board[position] != 0:
                    continue
                add_child_node(1, position, 0.9, current_node)
                add_child_node(2, position, 0.1, current_node)

            if(state_percentage!=1.0):
                if(len(current_node.children)%2 != 0):
                    raise Exception(f"len(current_node.children) = {len(current_node.children)},%2 != 0")
                chosen_length = math.ceil(len(current_node.children) * state_percentage)
                chosen_elements = random.sample(current_node.children, chosen_length)
                current_node.children = chosen_elements

        else:
            # "random after->state"
            children_position_lis = []
            for position in range(16):
                if current_node.state.board[position] != 0:
                    continue
                children_position_lis.append(position)
            children_position_len = len(children_position_lis)

            position_random = random.randint(0,children_position_len-1)
            child_position = children_position_lis[position_random]
            tile_random = random.random()
            if(tile_random >= 0.9):
                add_child_node(2, child_position, 1, current_node)
            else:
                add_child_node(1, child_position, 1, current_node)

    return current_node.children




def add_child_node(tile_value: int, position: int, possibility: float, node: exp_node):
    child_board_state = node.state.clone()
    child_board_state.board[position] = tile_value
    child_node = exp_node(state=child_board_state, height=node.height - 1,
                          type="state", possiblity=possibility)
    node.children.append(child_node)


def calculate_value(board: np.ndarray, model, device_number=None, batch_size = None) -> np.ndarray:
    # get value for board
    # board : np.array (seq_len,16)
    board_lenth = board.shape[0]

    input_boards = np.zeros((board_lenth, INPUT_SIZE), dtype="float32")

    for itr in range(board_lenth):
        model.make_input(input_boards[itr, :], board[itr])
    if(device_number!=None):
        answer = model.predict(input_boards,device_number, batch_size= batch_size).cpu().detach()
    else:
        answer = model.predict(input_boards,batch_size= batch_size).cpu().detach()
    return answer

def quick_calculate(board: np.ndarray, model, device_number=None, batch_size = None) -> np.ndarray:

    # get quick_value for board
    # board : np.array (seq_len,16)
    board_length = board.shape[0]
    input_boards = np.zeros((board_length, INPUT_SIZE), dtype="float32")

    for itr in range(board_length):
        model.make_input(input_boards[itr, :], board[itr])

    if torch.is_tensor(input_boards) == False:
        input_boards = torch.from_numpy(input_boards).float()

    with torch.no_grad():
        if device_number is not None:
            if device_number == -1:
                device = torch.device(f"cpu")
            else:
                device = torch.device(f"cuda:{device_number}")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_boards = input_boards.to(device)
        answer = model.predict(input_boards,device_number, batch_size = batch_size)
        answer = answer.cpu().detach()
    # return answer.numpy()
    return answer


def get_node_exp(current_node: exp_node, discount_factor=1, random_state=False, greedy_move=False, normalized = False, average_score = 1, state_percentage = 1.0) -> float:
    if (current_node.game_over == True):
        if (current_node.value == -float("inf")):
            raise Exception(f"One end node value is not calculated at height {current_node.height}")
        # current_node.exp = current_node.value
        # check this
        current_node.exp = discount_factor * current_node.value
        return current_node.exp

    if (current_node.type == "after_state"):
        children_exp_reward_lis = []
        for child_node in current_node.children:
            children_exp_reward_lis.append(child_node.possibility * (get_node_exp(child_node, discount_factor, random_state, greedy_move, normalized, average_score, state_percentage =state_percentage)))
        if not random_state:

            if(state_percentage == 1.0):
                current_node.exp = discount_factor * 2 * sum(children_exp_reward_lis) / len(children_exp_reward_lis)
            else:
                possibility_sum = sum( child_node.possibility for child_node in current_node.children )
                current_node.exp = discount_factor  * sum(children_exp_reward_lis) / possibility_sum

        else:
            current_node.exp = discount_factor * sum(children_exp_reward_lis) / len(children_exp_reward_lis)
        return current_node.exp

    if (current_node.type == "state"):
        best_act = -1
        best_node = None
        best_value = -float("inf")
        for itr, child_node in enumerate(current_node.children):
            if(normalized):
                children_exp_reward = get_node_exp(child_node, discount_factor, random_state, greedy_move, normalized, average_score, state_percentage =state_percentage) + (child_node.state.score - current_node.state.score)/average_score
            else:
                children_exp_reward = get_node_exp(child_node, discount_factor, random_state, greedy_move, normalized, average_score, state_percentage =state_percentage) + child_node.state.score - current_node.state.score

            if(greedy_move):
                #return greedy value.
                if(child_node.value == -float("Inf")):
                    raise Exception("simple move but deeper than 3")

                local_reward = child_node.state.score - current_node.state.score
                if (normalized):
                    local_reward = local_reward/average_score

                children_exp_reward = local_reward + child_node.value

            if (children_exp_reward > best_value):
                best_act = current_node.available_actions[itr]
                best_value = children_exp_reward
                best_node = child_node
            # print(children_exp_reward, ",", best_act,",",best_value)

        current_node.next_move = best_act
        current_node.next_node = best_node
        # print("best_act:",best_act)
        current_node.exp = best_value
        return current_node.exp


def get_double_node_exp(current_node: exp_node, discount_factor=1, random_state=False, greedy_move=False, normalized = False, average_score = 1) -> float:
    #only used in double learning.
    if (current_node.game_over == True):
        if (current_node.double_value == -float("inf")):
            raise Exception(f"One end node value is not calculated at height {current_node.height}")
        current_node.double_exp = discount_factor * current_node.double_value
        return current_node.double_exp

    if (current_node.type == "after_state"):
        children_exp_reward_lis = []
        for child_node in current_node.children:
            children_exp_reward_lis.append(child_node.possibility * (get_double_node_exp(child_node, discount_factor, random_state, greedy_move, normalized, average_score)))
        if not random_state:
            current_node.double_exp = discount_factor * 2 * sum(children_exp_reward_lis) / len(children_exp_reward_lis)
        else:
            current_node.double_exp = discount_factor * sum(children_exp_reward_lis) / len(children_exp_reward_lis)
        return current_node.double_exp

    if (current_node.type == "state"):

        best_next_node = current_node.next_node



        if(normalized):
            exp_reward = get_double_node_exp(best_next_node, discount_factor, random_state,
                                             greedy_move, normalized, average_score) + (best_next_node.state.score - current_node.state.score)/ average_score
        else:
            exp_reward = get_double_node_exp(best_next_node, discount_factor, random_state, greedy_move, normalized, average_score) + best_next_node.state.score - current_node.state.score
        current_node.double_exp = exp_reward
        return current_node.double_exp


def expand_and_get(root_state: State, model, height=3, greedy_value=False, return_node=False,
                   greedy_move=False, random_state= False, discount_factor=1, double_learning=False,
                   value_model=None, device_number=None, quick = False, normalized = False, average_score = 1,
                   batch_size = None, RC = False, state_percentage = 1.0, speed_up = False
                   ):
    """
    :param state_percentage: update x% states
    :param RC: Used to speed up with "greedy_move"
    :param batch_size: Only used to limit GPU memory cost in "3-ply"
    :param quick: Try to speed up. But it doesn't work.
    :param device_number:
    :param root_state:
    :param model:
    :param height:
    :param greedy_value:
    :param return_node:
    :param greedy_move:
    :param random_state:
    :param discount_factor:
    :param double_learning: Use double learning or not. If so, we use "model" to select move a*, use value model to get
        value from move a*.
    :param value_model:  Only used in double_learning.
    :return:
    """
    if (value_model != None):
        if(double_learning == False):
            raise Exception(f"double_learning setting wrong")

    root_node = exp_node(state=root_state.clone(), height=height)
    standby_node_lis = [root_node]
    bottom_node_lis = []
    bottom_board_lis = []

    # if(greedy_move == True) and (RC == False):
    if False == True:

        # We try to speed up greedy_move with less computation. However, because of parallel from GPU, it even slows down the training/generating.
        # it doesn't work for "RC" （renew all children）
        pass

        # while (standby_node_lis != []):
        #     children_node_lis = []
        #     for node in standby_node_lis:
        #         children_node_lis.extend(expand(node, bottom_nodes=bottom_node_lis, random_state=random_state, state_percentage =state_percentage))
        #     standby_node_lis = children_node_lis
        #
        # bottom_node_lis = []
        # for cildren_node in root_node.children:
        #     bottom_node_lis.append(cildren_node)
        #     bottom_board_lis.append(cildren_node.state.board)
        #
        #
        # bottom_board_array = np.array(bottom_board_lis)
        # if (quick):
        #     bottom_answer_array = quick_calculate(bottom_board_array, model, device_number=device_number, batch_size=batch_size)
        # else:
        #     bottom_answer_array = calculate_value(bottom_board_array, model, device_number=device_number, batch_size=batch_size)
        # for itr, bottom_node in enumerate(bottom_node_lis):
        #     bottom_node.value = bottom_answer_array[itr].item()
        #
        # best_act = -1
        # best_value = -float("inf")
        # best_node = None
        # for itr, child_node in enumerate(root_node.children):
        #
        #     local_reward = child_node.state.score - root_node.state.score
        #     if (normalized):
        #         local_reward = local_reward / average_score
        #
        #     children_exp_reward = discount_factor * child_node.value + local_reward
        #     if (children_exp_reward > best_value):
        #         best_act = root_node.available_actions[itr]
        #         best_value = children_exp_reward
        #         best_node = child_node
        # root_node.next_node = best_node
        # root_node.next_move = best_act
        # standby_node_lis = [root_node.next_node]
        # bottom_node_lis = []
        # bottom_board_lis = []
        #
        # while (standby_node_lis != []):
        #     children_node_lis = []
        #     for node in standby_node_lis:
        #         children_node_lis.extend(expand(node, bottom_nodes=bottom_node_lis, random_state=random_state, state_percentage =state_percentage))
        #     standby_node_lis = children_node_lis
        # # calculate bottom_value
        # for bottom_node in bottom_node_lis:
        #     bottom_board_lis.append(bottom_node.state.board)
        #
        # bottom_board_array = np.array(bottom_board_lis)
        # if (quick):
        #     bottom_answer_array = quick_calculate(bottom_board_array, model, device_number=device_number, batch_size=batch_size)
        # else:
        #     bottom_answer_array = calculate_value(bottom_board_array, model, device_number=device_number, batch_size=batch_size)
        # for itr, bottom_node in enumerate(bottom_node_lis):
        #     bottom_node.value = bottom_answer_array[itr].item()
        #
        # # child_ev = get_node_exp(current_node=root_node.next_node, discount_factor=discount_factor,
        # #                         random_state=random_state, greedy_move=greedy_move, normalized=normalized,
        # #                         average_score=average_score, state_percentage =state_percentage)
        # # The up is the previous version for child_ev. It doesn't have problem in 2-ply search, but has problem in 3-ply search or more.
        #
        # child_ev = get_node_exp(current_node=root_node.next_node, discount_factor=discount_factor, random_state=random_state, greedy_move=False, normalized=normalized, average_score= average_score, state_percentage =state_percentage)
        #
        # local_reward = root_node.next_node.state.score - root_node.state.score
        # if (normalized):
        #     local_reward = local_reward / average_score
        #
        # root_node.exp = child_ev + local_reward
        # if (return_node == False):
        #     return root_node.next_move, root_node.exp
        # else:
        #     return root_node.next_move, root_node.exp, root_node

    else:
    # expand
        while (standby_node_lis != []):
            children_node_lis = []
            for node in standby_node_lis:
                children_node_lis.extend(expand(node, bottom_nodes=bottom_node_lis, random_state=random_state, state_percentage =state_percentage))
            standby_node_lis = children_node_lis

        bottom_board_lis = []
        # calculate bottom_value
        for bottom_node in bottom_node_lis:
            bottom_board_lis.append(bottom_node.state.board)

        if greedy_value or return_node :
            # calculate Value(children)
            for cildren_node in root_node.children:
                bottom_node_lis.append(cildren_node)
                bottom_board_lis.append(cildren_node.state.board)

        bottom_board_array = np.array(bottom_board_lis)
        if (quick):
            bottom_answer_array = quick_calculate(bottom_board_array, model, device_number=device_number, batch_size=batch_size)
        else:
            bottom_answer_array = calculate_value(bottom_board_array, model, device_number=device_number, batch_size=batch_size)
        for itr, bottom_node in enumerate(bottom_node_lis):
            bottom_node.value = bottom_answer_array[itr].item()

        # print("***:",len(root_node.children) )
        ev = get_node_exp(current_node=root_node, discount_factor=discount_factor,random_state=random_state, greedy_move=greedy_move, normalized=normalized, average_score=average_score, state_percentage =state_percentage)

        if(double_learning == False):
            if greedy_value:
                if (return_node == False):
                    return root_node.next_move, root_node.next_node.state.score - root_node.state.score + discount_factor * root_node.next_node.value
                else:
                    return root_node.next_move, root_node.next_node.state.score - root_node.state.score + discount_factor * root_node.next_node.value, root_node
            elif greedy_move:

                best_act = -1
                best_value = -float("inf")
                best_node = None
                for itr, child_node in enumerate(root_node.children):
                    if(normalized):
                        children_exp_reward = discount_factor * child_node.value + (child_node.state.score - root_node.state.score)/average_score
                    else:
                        children_exp_reward = discount_factor * child_node.value + child_node.state.score - root_node.state.score
                    if (children_exp_reward > best_value):
                        best_act = root_node.available_actions[itr]
                        best_value = children_exp_reward
                        best_node = child_node
                    # print(children_exp_reward, ",", best_act,",",best_value)
                if (return_node == False):
                    return best_act, best_node.exp
                else:
                    return best_act, best_node.exp, root_node

            else:
                if (return_node == False):
                    return root_node.next_move, ev
                else:
                    return root_node.next_move, ev, root_node
        else:
            # Use it in double_learning
            standby_node_lis = [root_node]
            bottom_node_lis = []
            while (standby_node_lis != []):
                children_node_lis = []
                for node in standby_node_lis:
                    if node.game_over and node.double_value == -float("inf"):
                        bottom_node_lis.append(node)
                    if node.type == "state":
                        if(node.next_node != None):
                            children_node_lis.append(node.next_node)
                    else:
                        children_node_lis.extend(node.children)
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
            if (quick):
                bottom_answer_array = quick_calculate(bottom_board_array, value_model, device_number=device_number)
            else:
                bottom_answer_array = calculate_value(bottom_board_array, value_model, device_number=device_number, batch_size=batch_size)
            for itr, bottom_node in enumerate(bottom_node_lis):
                bottom_node.double_value = bottom_answer_array[itr].item()
            double_ev = get_double_node_exp(current_node=root_node, discount_factor=discount_factor,random_state=random_state, greedy_move=greedy_move, normalized=normalized, average_score=average_score)

            if greedy_value:
                local_reward = root_node.next_node.state.score - root_node.state.score
                if (normalized):
                    local_reward = local_reward / average_score
                return root_node.next_move, local_reward + discount_factor * root_node.next_node.double_value, root_node
            elif greedy_move:

                best_act = -1
                best_value = -float("inf")
                best_node = None
                for itr, child_node in enumerate(root_node.children):
                    local_reward = child_node.state.score - root_node.state.score
                    if (normalized):
                        local_reward = local_reward / average_score
                    children_exp_reward = discount_factor * child_node.value + local_reward
                    if (children_exp_reward > best_value):
                        best_act = root_node.available_actions[itr]
                        best_value = children_exp_reward
                        best_node = child_node
                    # print(children_exp_reward, ",", best_act,",",best_value)
                return best_act, best_node.double_exp, root_node

            else:
                return root_node.next_move, double_ev, root_node



if __name__ == '__main__':
    board = np.ones((2, 16), dtype="int")
    input_boards = np.zeros((board.shape[0], INPUT_SIZE), dtype="float32")
    # for itr in range(2):
    # make_input(input_boards[itr,:],board[itr,:])
    print(input_boards)
