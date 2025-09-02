import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as opt
checkpointprefix = 'weights_1'



class Model(nn.Module):
    def __init__(self, learning_rate = 0.001):
        super(Model, self).__init__()
        self.DIM_I = 416
        self.DIM_O = 1
        self.description = 'FC + Two Conv2D (2x2) layers and three full-connect layers. Number of outputs of each layer is set 256, 512, 1024, 256, 1.'
        self.numparams = 1

        self.fc1 = nn.Linear(416,32)
        self.fc2 = nn.Linear(32,128)

        self.conv1 = nn.Conv2d(26,256, kernel_size= (2,2),padding=(0,0), stride= (1,1))
        self.conv2 = nn.Conv2d(256,512, kernel_size=(2, 2),padding=(0,0), stride=(1, 1))
        self.flat1 = nn.Flatten()
        self.Linear1 = nn.Linear(2048, 1024)
        self.Linear2 = nn.Linear(1024, 256)
        self.Linear3 = nn.Linear(256, 1)
        # self.MSEloss = nn.MSELoss()
        self.optimizer = opt.Adam(self.parameters(), lr= learning_rate)
        self.output_v = None

    def forward (self,x):
        x1 = torch.reshape(x,(-1,26,4,4))

        xf1 = nn.Flatten()(x1)
        xf2 = F.relu(self.fc1(xf1))
        xf3 = F.relu(self.fc2(xf2))
        xf4 = torch.reshape(xf3,(-1,8,4,4) )

        x_value = x1[:, :18, :, :]
        global_in = torch.cat((x_value,xf4),1 )

        x2 = F.relu(self.conv1(global_in))
        x3 = F.relu(self.conv2(x2))
        x4 = self.flat1(x3)
        x5 = F.relu(self.Linear1(x4))
        x6 = F.relu(self.Linear2(x5))
        x7 = self.Linear3(x6)
        self.output_v = x7
        return x7
    def num_flat_features(self, x):

        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def make_input(self,x,board):
        for j in range(16):
            b = board[j]
            x[16 * b + j] = 1
            x[16 * (18 + int(j//4) ) + j ] = 1
            x[16 * (22 + j%4) + j  ] = 1

    # def predict(self, x, device_number=None):
    #
    #     if(torch.is_tensor(x) == False):
    #         x = torch.from_numpy( x )
    #     # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    #     # gpux = torch.tensor(x,device=device)
    #
    #     if(device_number == None):
    #         x = x.cuda()
    #     elif(device_number == -1):
    #         device = torch.device(f"cpu")
    #         x = x.to(device)
    #     else:
    #         device = torch.device(f"cuda:{device_number}")
    #         x = x.to(device)
    #
    #     ans = self.forward(x)
    #     return ans

    def predict(self, x, device_number=None, batch_size=None):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        if device_number is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_number == -1:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{device_number}')

        x = x.to(device)

        if batch_size is None or batch_size >= len(x):
            ans = self.forward(x)
        else:
            # 分批处理数据
            ans = []
            for start in range(0, len(x), batch_size):
                end = start + batch_size
                x_batch = x[start:end]
                ans_batch = self.forward(x_batch)
                ans.append(ans_batch)

            # 合并结果
            ans = torch.cat(ans, dim=0)

        return ans

    def quick_predict(self,x):
        # device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        # gpux = torch.tensor(x,device=device)
        x.cuda()
        ans = self.forward(x)
        return ans


    def train_mode(self,x_train,y_train, batch_size=1024, device_number=None):
        if(torch.is_tensor(x_train) == False):
            x_train = torch.from_numpy( x_train )
        if(torch.is_tensor(y_train) == False):
            y_train = torch.from_numpy( y_train )
        if(device_number == None):
            x_train = x_train.cuda()
            y_train = y_train.cuda()
        elif(device_number == -1):
            device = torch.device(f"cpu")
            x_train = x_train.to(device)
            y_train = y_train.to(device)
        else:
            device = torch.device(f"cuda:{device_number}")
            x_train = x_train.to(device)
            y_train = y_train.to(device)

        ans1 = self.forward(x_train)
        self.optimizer.zero_grad()
        loss_maker = nn.MSELoss()
        loss1 = loss_maker(ans1,y_train)
        loss1.backward()
        self.optimizer.step()
        return loss1

def freeze(freeze_model: Model):
    for param in freeze_model.fc1.parameters():
        param.requires_grad = False
    for param in freeze_model.fc2.parameters():
        param.requires_grad = False
    for param in freeze_model.conv1.parameters():
        param.requires_grad = False
    for param in freeze_model.conv2.parameters():
        param.requires_grad = False


net= Model()
INPUT_SIZE = net.DIM_I
# net.load_state_dict(torch.load(checkpointprefix) )

def restore_mod ( checkpointprefix ):

    global net
    net.load_state_dict(torch.load(checkpointprefix))


def policy_value( state_batch):
    """
    input: a batch of states
    output: a batch of action probabilities and state values
    """
    value = net.predict(state_batch).cpu().detach().numpy()
    return value

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def testcnn():
    net = Model().cuda(0)
    input1 = np.zeros( (2,416,),dtype="float32" )
    inputp = torch.from_numpy(input1)
    # inputp = torch.randn( (2,256),dtype="float32" )
    ans1 = net.predict(inputp)
    print( ans1 )
    True_ans = np.array( [ [1.0],[1.0] ] ,dtype="float32")
    print(True_ans.dtype == "float32" )
    # True_ans = torch.from_numpy( True_ans )
    print(True_ans.dtype)
    print(count_parameters(net))
    print(net.predict(input1).cpu().detach()[0].item())


    # loss_maker = nn.MSELoss()
    # loss1 = loss_maker(ans1,True_ans)
    # optimizer = opt.Adam(net.parameters(),lr=0.001)
    # optimizer.zero_grad()
    # loss1.backward()
    # optimizer.step()
    l = net.train_mode(x_train=inputp,y_train= True_ans)
    l = net.train_mode(x_train=inputp,y_train= True_ans)

    # ans1 = net.predict(inputp)
    # print( ans1 ,l)
#
# def test2():
#     x = np.zeros([4, model.DIM_I], dtype="float32")

def test_save():
    net = Model().cuda()
    input1 = np.zeros( (2,416,),dtype="float32" )
    inputp = torch.from_numpy(input1)

    ans1 = net.predict(inputp)
    print( ans1 )
    torch.save(net.state_dict(),"test_weights")

def test3():
    input1 = np.zeros((3, 416,), dtype="float32")
    inputp = torch.from_numpy(input1)
    ans1 = policy_value(inputp)
    print(ans1)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# # Assuming 'model' is an instance of 'Model'
# print("Total number of parameters: ", count_parameters(model))

if __name__ == '__main__':

    testcnn()
    # print(y)
    # test3()
    # test_save()



