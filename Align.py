import cv2
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

m=7.0  #n^2-1
Maxpool = nn.MaxPool2d(kernel_size=(3,3),stride=1,padding=1)
Relu = nn.ReLU()
One_pad = torch.nn.ReplicationPad2d(1)
# Neighbour kernel
one = torch.tensor(np.ones([1,1,3,3])).type(torch.float32)
one[0][0][1][1]=torch.tensor([0.])

def Readimg(imgpath):
    img = cv2.imread(imgpath)   #read img ,already a silhouette
    w,h = img.shape[:-1]  #w and h
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #to grey
    ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)   ## to binary {0,1}
    S0=binary
    S0 = torch.from_numpy(S0).type(torch.float32)
    S0 = S0.unsqueeze(0)
    S0 = S0.unsqueeze(0)
    return (S0,w,h)



def S2D(S0,w,h):
    #init Din
    Din=np.zeros([S0.shape[2],S0.shape[3]])
    # cv2.imwrite('S0.jpeg',S0[0][0].detach().numpy()*255)
    # pdb.set_trace()
    Din = Din + S0[0][0].detach().numpy()
    Rout1 = Relu(F.conv2d(S0,one,stride=1,padding=1)-m)
    Din = Din + Rout1[0][0].detach().numpy()
    # cv2.imwrite('S1.jpeg',Rout1[0][0].detach().numpy()*255)
    # iter till all black, cal Din , equation 2
    for iiii in range(400):
        Rout1 = Relu(F.conv2d(Rout1,one,stride=1,padding=1)-m)
        # cv2.imwrite('./tmp/S'+ str(iiii) +'.jpeg',Rout1[0][0].detach().numpy()*255)
        Din = Din + Rout1[0][0].detach().numpy()
        if Rout1.max()<1:
            # print('Finish Din!')
            # cv2.imwrite('Din.jpeg',Din)
            break
    return Din


def S2D_reverse(S0,w,h):
    # differnce is the PADDING needed to be corret to ONE!  11111111111111111111
    #init Din
    Din=np.zeros([S0.shape[2],S0.shape[3]])
    # cv2.imwrite('S0.jpeg',S0[0][0].detach().numpy()*255)
    # pdb.set_trace()
    Din = Din + S0[0][0].detach().numpy()
    S0 = One_pad(S0)
    Rout1 = Relu(F.conv2d(S0,one,stride=1)-m)
    Din = Din + Rout1[0][0].detach().numpy()
    # cv2.imwrite('S1.jpeg',Rout1[0][0].detach().numpy()*255)
    # iter till all black, cal Din , equation 2
    for iiii in range(400):
        Rout1 = One_pad(Rout1)
        Rout1 = Relu(F.conv2d(Rout1,one,stride=1)-m)
        # cv2.imwrite('./tmp/S'+ str(iiii) +'.jpeg',Rout1[0][0].detach().numpy()*255)
        Din = Din + Rout1[0][0].detach().numpy()
        if Rout1.max()<1:
            # print('Finish Din!')
            # cv2.imwrite('Din.jpeg',Din)
            break
    return Din



def D2T(Din,S0):
    # cal T, equation 3
    # print(Din.shape)
    MpDin = Maxpool(torch.from_numpy(Din.reshape(1,1,Din.shape[0],Din.shape[1])))
    Din = Relu(torch.from_numpy(Din) - MpDin[0][0]  +1)
    T_real = Din*S0[0][0]
    # cv2.imwrite('T.jpeg',np.array(Din)*255)
    # cal DT_out for eq5
    return T_real

#Assume this is the predict 
(S0_1,w_1,h_1) = Readimg('S.jpg')
Din_1 = S2D(S0_1,w_1,h_1)
T_1 = D2T(Din_1,S0_1)
T_1 = T_1.type(torch.float32)
T_1 = T_1.unsqueeze(0)
T_1 = T_1.unsqueeze(0)
cv2.imwrite('1_T.jpeg',np.array((1-T_1)[0][0])*255)
DT_out_1 = S2D_reverse(1-T_1,w_1,h_1)
T_1 = np.array(T_1[0][0])
cv2.imwrite('DT_out_1.jpeg',DT_out_1)

#Assume this is the GT
(S0_2,w_2,h_2) = Readimg('S2.jpg')
Din_2 = S2D(S0_2,w_2,h_2)
T_2 = D2T(Din_2,S0_2)
T_2 = T_2.type(torch.float32)
T_2 = T_2.unsqueeze(0)
T_2 = T_2.unsqueeze(0)
cv2.imwrite('2_T.jpeg',np.array((1-T_2)[0][0])*255)
DT_out_2 = S2D_reverse(1-T_2,w_2,h_2)
T_2 = np.array(T_2[0][0])
cv2.imwrite('DT_out_2.jpeg',DT_out_2)

#unfortunately , this two T, not have same shape, we just can select part of it
# (313,288) (335,357)
LTsp1 = T_1[11:335-11,33:357-36]*DT_out_2
cv2.imwrite('LTsp1.jpeg',LTsp1/LTsp1.max()*255)
LTsp2 = DT_out_1[11:335-11,33:357-36]*T_2
cv2.imwrite('LTsp2.jpeg',LTsp2/LTsp2.max()*255)
LT_sp = LTsp1.sum()+LTsp2.sum()
print('LT_sp:',LT_sp)

#then we show that this is not spatial invariable, of course, Chamfer distance does not have this property too, is there any Distance measure that is spatial invariable?
LTsp1 = T_1[0:335-22,0:357-69]*DT_out_2
cv2.imwrite('LTsp1_offset.jpeg',LTsp1/LTsp1.max()*255)
LTsp2 = DT_out_1[0:335-22,0:357-69]*T_2
cv2.imwrite('LTsp2_offset.jpeg',LTsp2/LTsp2.max()*255)
LT_sp = LTsp1.sum()+LTsp2.sum()
print('LT_sp_offset:',LT_sp)

