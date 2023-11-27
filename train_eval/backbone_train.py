
# Code in cnn_fn_pytorch.py
from __future__ import print_function, division
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import argparse
from time import time
from torch.optim import lr_scheduler
from torchsummary import summary
from torch.utils.data import *
from imutils import paths
from tqdm import tqdm

sys.path.append(".") 
from model import *
from utils import *
from torch.optim import lr_scheduler
from torchsummary import summary

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images",    default='/data2/user/dwwang3/workspace/github.com/qrcode/listenai_qrcode/datasets/data/train_data/merge', help="input file path")
ap.add_argument("-n", "--epochs",    default=100,  help="epochs for train")
ap.add_argument("-b", "--batchsize", default=64,  help="batch size for train")
ap.add_argument("-r", "--resume",    default='./weights/wR2/wR2.pth4', help="file for re-train")
ap.add_argument("-w", "--writeFile", default='wR2.out', help="file for output")
args = vars(ap.parse_args())


import datetime
import wandb
wandb.login(key='4ec0396ddf4a239b6fcb4daa9e15710b5cf963cd')
# wandb = None

numClasses = 4  
# imgSize = (480, 480)
imgSize = (240, 320)

def train_backbone():

    wandb.init(
            # set the wandb project where this run will be logged
            project="vits",
            config = args,
            name = 'orginal' + datetime.datetime.now().strftime('%F %T')
        )

    batchSize = int(args["batchsize"])
    modelFolder = 'weights/wR2/'
    storeName = modelFolder + 'wR2.pth'
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
    print("Using {} device training.".format(device.type))
    print(device.type)

    num_epochs = int(args["epochs"])
    #   initialize the output file
    with open(args['writeFile'], 'wb') as outF:
        pass

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp


    model = wR2_1(numClasses)
    epoch_start = 0
    resume_file = str(args["resume"])
    # if not resume_file == '111':
    if resume_file == '111':
        # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
        if not os.path.isfile(resume_file):
            print ("fail to load existed model! Existing ...")
            exit(0)
        model.load_state_dict(torch.load('./weights/wR2/wR2.pth4', map_location='cpu'))

    if use_gpu:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = model.cuda()
    else:
        model.to(device)
    print(model)
    print(get_n_params(model))

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lrScheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # dst = LocDataLoader([args["images"]], imgSize)
    dst = ChaLocDataLoader(args["images"].split(','), imgSize)
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)

    # since = time.time()
    for epoch in range(epoch_start, num_epochs):
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()

        train_bar = tqdm(trainloader, file=sys.stdout)
        for i, (XI, YI) in enumerate(train_bar):
            # print('%s/%s %s' % (i, times, time()-start))
            YI = np.array([el.numpy() for el in YI]).T
            if use_gpu:
                x = Variable(XI.cuda(0))
                y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(YI), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Compute and print loss
            loss = 0.0
            if len(y_pred) == batchSize:
                loss += 0.8 * nn.L1Loss().cuda()(y_pred[:][:2], y[:][:2])
                loss += 0.2 * nn.L1Loss().cuda()(y_pred[:][2:], y[:][2:])
                # lossAver.append(loss.data[0])
                lossAver.append(loss)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.save(model.state_dict(), storeName)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, num_epochs, sum(lossAver) / len(lossAver))

            wandb.log( {"lossAver":lossAver, } )
            
            # if i % 5 == 1:
            #     print ('epoch:%s - index:%s - loss:%s' % (epoch, i, sum(lossAver) / len(lossAver)))
            #     with open(args['writeFile'], 'a') as outF:
            #         outF.write('train %s images, use %s seconds, loss %s' % (i*batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:])))
        
        # print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        with open(args['writeFile'], 'a') as outF:
            outF.write('Epoch: %s %s %s' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        torch.save(model.state_dict(), storeName + str(epoch))

        wandb.finish()


def eva_backbone():
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
    model = wR2_1(numClasses)
    model.to(device)
    model.load_state_dict(torch.load('./weights/wR2/wR2.pth4', map_location='cpu'))
    model.eval()

    # img_org = cv2.imread("./data_set/simple_test/test/01-89_84-231&482_413&539-422&537_242&536_219&475_399&476-0_0_15_11_25_26_26-121-16.jpg")
    # img_org = cv2.imread("./data_set/simple_test/test/01-90_86-222&446_405&507-400&505_225&500_227&442_402&447-0_0_20_0_24_30_26-66-28.jpg") 
    img_org = cv2.imread("./data_set/simple_test/test/01-90_87-245&456_432&532-445&528_267&529_251&471_429&470-0_0_10_4_28_24_32-54-11.jpg") 
    # img_org = cv2.imread("./data_set/simple_test/test/2.jpg")  

    org_h, org_w= img_org.shape[0], img_org.shape[1],
    img_resize = cv2.resize(img_org, imgSize)
    img_tesor = torch.from_numpy(img_resize)
    img_tesor = img_tesor.unsqueeze(dim=0)
    img_tesor = torch.permute(img_tesor, (0,3,1,2)).contiguous()
    img_tesor = img_tesor.float()
    img_tesor /= 255.0
    img_tesor = img_tesor.to(device)
    y_pred = model(img_tesor)
    print(y_pred)

    resized_w = imgSize[0]
    resized_h = imgSize[1]
    cw, ch, w, h = y_pred[0][0]*resized_w, y_pred[0][1]*resized_h, y_pred[0][2]*resized_w, y_pred[0][3]*resized_h, 

    leftup_x, leftup_y, rightdown_x, rightdown_y = cw - w/2, ch - h/2, cw + w/2, ch + h/2
    cv2.rectangle(img_resize, (int(leftup_x), int(leftup_y)), (int(rightdown_x), int(rightdown_y)), (0, 255, 255), 1 )
    cv2.imshow('image_resize', img_resize)
    cv2.waitKey(0)

    leftup_x, leftup_y, rightdown_x, rightdown_y = leftup_x*(org_w/resized_w), leftup_y*(org_h/resized_h), rightdown_x*(org_w/resized_w), rightdown_y*(org_h/resized_h) 
    cv2.rectangle(img_org, (int(leftup_x), int(leftup_y)), (int(rightdown_x), int(rightdown_y)), (0, 255, 255), 1 )
    cv2.imshow('image_org', img_org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
    # assert img.shape[0] == 1160
    # new_labels = [(leftUp[0] + rightDown[0])/(2*ori_w), (leftUp[1] + rightDown[1])/(2*ori_h), (rightDown[0]-leftUp[0])/ori_w, (rightDown[1]-leftUp[1])/ori_h]

if __name__ == '__main__':
    train_backbone()
    # eva_backbone()
