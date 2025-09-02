import get_init_value

model_names = [
    # ['1ply_l', '7'],
    # ['1ply_s', '7'],
    # ['1ply_l_s', '7'],
    #
    # ['2ply_double', '7'],
    # ['2ply_double_l', '7'],
    # ['2ply_double_l_s', '7'],
    # ['2ply_double_s', '7'],

    # ['2ply_double_separate', '7'],
    # ['2ply_double_separate_l', '7'],
    # ['2ply_double_separate_l_s', '7'],
    # ['2ply_double_separate_s', '7'],

    # ['2ply', '7'],
    # ['2ply_s', '7'],
    # ['2ply_l', '7'],
    # ['2ply_l_s', '7'],
    #
    # ['RC_renew_best_simple_move', '7'],
    # ['RC_best_l', '7'],
    # ['RC_best_l_s', '7'],
    # ['RC_best_s', '7'],
    #
    # ['RC_double', '7'],
    # ['RC_double_l', '7'],
    # ['RC_double_l_s', '7'],
    # ['RC_double_s', '7'],
    #
    # ['RC_double_separate', '7'],
    # ['RC_double_separate_l', '7'],
    # ['RC_double_separate_l_s', '7'],
    # ['RC_double_separate_s', '7'],
    #
    # ['RC_s', '7'],
    # ['RC', '7'],
    # ['RC_l', '7'],
    # ['RC_l_s', '7'],
    # ["2ply_from0", '7'],
    # ["2ply_double_separate_from0", '7'],
    # ["2ply_double_from0", '7'],
    # ["1ply_from0", '7'],
    # ["RC_best_from0", '7'],
    # ["D1_totest",'7'],
    ["2ply_TD_afterstate","1"],
    # ["2ply_l","7"],

]

for model_name in model_names:
    get_init_value.write_init_value(
        model_name=model_name[0],
        thread_name=model_name[1],

    )

