#scp -r miacp:./wangweikai/final_2048/CNN-2/program/exp/training_cnn22B_1/weights-300.data-00000-of-00001 ./
#scp -r miacp:./wangweikai/final_2048/CNN-2/program/exp/training_cnn22B_1/weights-300.index ./
#scp -r miacp:./wangweikai/final_2048/CNN-2/program/exp/training_cnn22B_1/weights-300.meta ./

#scp -r ./train_more.py miacp:./wangweikai/final_2048/CNN-2/program/exp/
#scp -r ./train_more_cnn22B_2 miacp:./wangweikai/final_2048/CNN-2/program/exp/
scp miacp:./wangweikai/pytorch_2048_3/FC_CNN_PO_deeper/program/exp/Large_batch/training_cnn22B_2/weights-1000 ./
