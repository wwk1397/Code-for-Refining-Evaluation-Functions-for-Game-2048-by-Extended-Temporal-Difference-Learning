import numpy as np

def read_file_chunks(filename, chunk_size=100000):
    with open(filename, "r", encoding="utf-8") as file:
        chunk = []
        for line in file:
            if "possible_move" not in line or "restart 0 " not in line:
            # if "possible_move" not in line:
                continue
            chunk.append(line)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


mod_name = "D3_OD_deep_no"
weight_sum = 35
list_diction = [[] for _ in range(weight_sum+1)]
test_file_name = f"./{mod_name}/training_cnn22B_1/testing.log"
file_lenth = 0
for chunk in read_file_chunks(test_file_name):
    file_lenth += len(chunk)
whole_itr = 0
for chunk in read_file_chunks(test_file_name):
    for itr,line in enumerate(chunk):
        dic_itr = int( (weight_sum*whole_itr)/file_lenth )
        # print(dic_itr,line.split(" "))
        list_diction[dic_itr].append(float(line.split(" ")[9]))
        whole_itr +=1

with open("log_summary3.txt", "w", encoding="utf-8") as out_file:
    for i in range(weight_sum):
        if len(list_diction[i]) == 0:
            out_file.write(f"{i} 0\n")
        else:
            out_file.write(f"{i} {np.mean(list_diction[i])}\n")
