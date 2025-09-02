import numpy as np

#Input of different weight has different "snake order", we should rotate and flip.
#In our program, we ......

def rotate(arr):
    return np.array([arr[3], arr[0], arr[1], arr[2]])

def flip(arr):
    return np.array([arr[0], arr[3], arr[2], arr[1]])

def get_versions(arr):
    versions = [arr]
    for _ in range(3):
        arr = rotate(arr)
        versions.append(arr)
    return versions + [flip(v) for v in versions]

def get_transform(arr_A, arr_B):
    # return the parameters we will use to transfer B into A
    for i, version in enumerate(get_versions(arr_B)):
        if np.array_equal(version, arr_A):
            if i < 4:
                return (4-i)%4, 0
            else:
                return (8-i)%4 , 1
    return "No transformation found"

def transform_2d(board,rot_num,flip_num):
    # rot 90 * rot_num in counterclockwise, flip (0 or 1) in axis 1. B -> A
    board_2d = board.reshape((4, 4))
    board_2d = np.rot90(board_2d,rot_num)
    if flip_num ==1:
        board_2d= np.flip(board_2d,axis=1)
    return board_2d.flatten()

def transform_back_2d(board,rot_num,flip_num):
    # rot 90 * rot_num in counterclockwise, flip (0 or 1) in axis 1. A -> B
    board_2d = board.reshape((4, 4))
    if flip_num ==1:
        board_2d= np.flip(board_2d,axis=1)
    board_2d = np.rot90(board_2d,-rot_num)
    return board_2d.flatten()


# Example usage:
def test1():
    A = np.array([1,3,2,4])
    # B = np.array([2,0,0,1])  # This is A rotated 90 degrees clockwise
    B = np.array([2,4,1,3])
    rot_num,flip_num = get_transform(A, B)
    print(rot_num,flip_num)

    rot_num,flip_num = get_transform(B, A)
    print(rot_num,flip_num)

def test2():
    matrix6 = np.array([0,0,0,4,0,0,0,3,0,0,0,2,0,0,0,1])
    matrix7 = np.array([1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print(transform_2d(matrix6,1,1))
    print(transform_2d(matrix7, 1, 1))


if __name__ == '__main__':
    # test1()
    test2()
