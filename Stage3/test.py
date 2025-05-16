import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.utils.data import DataLoader
from utils.graph import *
from utils.siScore_utils import *
from utils.parameters import *
import os
from itertools import permutations
import copy
from tqdm import tqdm
from backbone import BackboneModel

class TestDataset2(Dataset):
    def __init__(self, transform=None):
        # self.file_list = glob.glob('../data/'+args.img+'/*/*.png')
        self.file_list = glob.glob('../data/extra_granules/*/*.png')
        self.transform = transform        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        direc = path.split("/")[-2]
        name = path.split("/")[-1].split(".png")[0]
        name = direc + '*' + name
        image = io.imread(path) / 255.0

        if self.transform:
            image = self.transform(np.stack([image])).squeeze()

        return image, name

args = siScore_parser()
device_num = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用模型进行测试，对图片进行打分！
print('Loading Best Model...')
model_test = BackboneModel(pretrain=False, num_gpus=device_num)
state_dict = torch.load('../checkpoint/{}_acc_0.8734_loss_0.1335'.format(args.name), weights_only=True)
print("Model information: loss: %.4f, acc: %.4f" % (state_dict['loss'], state_dict['acc']))
model_test.load_state_dict(state_dict['model'], strict=True)
model_test.to(device)
print('Finish Loading Model!')

model_test.eval()    
test_dataset = TestDataset2(transform = transforms.Compose([
                                        #transforms.ToTensor(),  
                                        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ToTensor(),  
                                        Grayscale(),
                                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4)
print(len(test_dataset))    
df2 = pd.DataFrame(columns = ['direc','y_x', 'score'])
cnt = 0

with torch.no_grad():
    testloader = tqdm(test_loader)
    for data, name in testloader:
        data = data.to(device)
        scores = model_test(data).squeeze()
        count = 0
        # scores = torch.clamp(scores, min=0, max=1)
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        for each_name in name:
            dist = each_name.split('*')[0]
            na  = each_name.split('*')[1]
            df2.loc[cnt] = [dist, na, scores[count].cpu().data.numpy()]
            cnt += 1  
            count += 1  
        df2.to_csv(args.img+'_scores.csv')
df2.to_csv(args.img+'_scores.csv')