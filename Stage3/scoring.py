import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.utils.data import DataLoader, random_split
from utils.graph import *
from utils.siScore_utils import *
from utils.parameters import *
import os
from itertools import permutations
import copy
from backbone import *
from tqdm import tqdm
# from Ranger.ranger import Ranger 

class TestDataset(Dataset):
    def __init__(self, transform=None):
        # args.img = 'NK'
        # 在 glob 模式中，* 是一个特殊的通配符，用于匹配任意数量的字符，但不包括路径分隔符（例如 /）。
        # self.file_list = glob.glob('../data/'+args.img+'/*/*.png')
        self.file_list = glob.glob('../data/'+args.img+'/*/*/*.png')
        self.transform = transform        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        #con = path.split("/")[-3][:3]
        direc = path.split("/")[-2]
        #name = path[-19:-9]
        name = path.split("/")[-1].split(".png")[0]
        name = direc + '*' + name
        #image =  Image.open(path)
        image = io.imread(path) / 255.0


        if self.transform:
            #image =  self.transform(image)
            image = self.transform(np.stack([image])).squeeze()

        return image, name

class TestDataset2(Dataset):
    def __init__(self, transform=None):
        # args.img = 'NK'
        # 在 glob 模式中，* 是一个特殊的通配符，用于匹配任意数量的字符，但不包括路径分隔符（例如 /）。
        self.file_list = glob.glob('../data/'+args.img+'/*/*.png')
        self.transform = transform        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        #con = path.split("/")[-3][:3]
        direc = path.split("/")[-2]
        #name = path[-19:-9]
        name = path.split("/")[-1].split(".png")[0]
        name = direc + '*' + name
        #image =  Image.open(path)
        image = io.imread(path) / 255.0


        if self.transform:
            #image =  self.transform(image)
            image = self.transform(np.stack([image])).squeeze()

        return image, name

def make_data_loader(cluster_list, batch_sz):
    cluster_dataset = ClusterDataset(cluster_list, 
                                     dir_name = args.dir_name, 
                                     transform = transforms.Compose([
                                       RandomRotate(),
                                       ToTensor(),
                                       Grayscale(prob = 1.0),
                                       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]))
    # cluster_loader = torch.utils.data.DataLoader(
    #     cluster_dataset, 
    #     batch_size=batch_sz, 
    #     shuffle=True, 
    #     num_workers=4, 
    #     drop_last=True
    # )
    # 对cluster_dataset作8:2分割
    train_size = int(0.8 * len(cluster_dataset))
    val_size = len(cluster_dataset) - train_size
    train_dataset, test_dataset = random_split(cluster_dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_sz, 
        shuffle=True, 
        num_workers=4, 
        drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_sz, 
        shuffle=False, 
        num_workers=4, 
        drop_last=False
    )
    return train_loader, val_loader

def get_cluster_target_scores(partial_order, cluster_unify):
    cluster_rank = []
    for pair in partial_order:
        if [pair[1]] not in cluster_rank:
            cluster_rank.append([pair[1]])
        cluster_rank.append([pair[0]])
    # 现在cluster_rank中包含了所有的cluster排序，从左到右表示发展水平由低到高
    # 接下来将cluster_unify中的cluster对应到cluster_rank中
    # 比如cluster_rank是[[1], [0], [5], [14], [17], [13], [12], [19]]
    # cluster_unify是[[1, 2, 3, 4, 6], [0, 7], [5, 8, 9, 10], [14, 15, 16, 20], [17, 21], [13, 18], [11, 12]]
    # 那么我希望cluster_rank变为[[1, 2, 3, 4, 6], [0, 7], [5, 8, 9, 10], [14, 15, 16, 20], [17, 21], [13, 18], [11, 12], [19]]
    for rank in cluster_rank:
        for cluster in cluster_unify:
            if rank[0] in cluster:
                # 替换原本的rank为cluster
                cluster_rank[cluster_rank.index(rank)] = cluster
                break
    
    # 接下来为cluster_rank里面每个簇赋分数，最低的簇赋0分，最高的簇赋1分
    # 从左到右依次赋分
    print(f'cluster_rank: {cluster_rank}')
    cluster_target_scores = {}
    for i in range(len(cluster_rank)):
        for j in range(len(cluster_rank[i])):
            cluster_target_scores[cluster_rank[i][j]] = 1- i/(len(cluster_rank)-1)

    return cluster_target_scores

def deactivate_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            m.eval()
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.zero_()


def train_and_eval(args, epoch, model, optimizer, scheduler, 
          train_loader,
          val_loader,
          target_scores,
          device):
    """
    训练模型。

    Args:
        args: 命令行参数。
        epoch (int): 当前的训练轮数。
        model (nn.Module): 要训练的模型。
        optimizer (Optimizer): 优化器。
        scheduler (LR_Scheduler): 学习率调度器。
        loader_list (dict): 聚类编号到数据加载器的映射字典。
        cluster_path_list (list of lists): 所有聚类路径的列表。
        device (torch.device): 设备（CPU 或 GPU）。
        target_scores (dict): 聚类编号到目标得分的映射字典。

    Returns:
        float: 当前轮的平均损失。
    """
    # backbone model是接了一个nn.Linear(512,1)的ResNet18
    # loader_list = {0: DataLoader, 1: DataLoader, ..., 22: DataLoader}
    model.train()
    # Deactivate the batch normalization before training
    # deactivate_batchnorm(model.module)
    deactivate_batchnorm(model)
    # 这些记录器用于统计训练过程中的平均损失值，方便监控模型的训练情况。

    # For each cluster route, train the model
    train_loss = 0
    train_acc = 0
    count = 0

    #print(scheduler.get_lr())
    train_loader = tqdm(train_loader, desc='Training', ncols=130)
    for imgs, labels in train_loader:
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # loss_reg
        # 将拼接的数据输入模型，得到预测分数scores，并将其裁剪到[0,1]范围。
        scores = model(imgs).squeeze()  # shape: [batch_size * cluster_num]
        # print(f'before clamping: {scores}')
        # scores = torch.clamp(scores, min=0, max=1)
        # print(f'after clamping: {scores}')
        target = torch.tensor([target_scores[label.item()] for label in labels]).to(device).float()
        # print(f'target: {target}')
        # loss_reg = nn.MSELoss()(scores, target)
        loss_reg = CombinedLoss()(scores, target)
        acc_reg = torch.sum(torch.abs(scores - target) < 0.07).item() / len(scores)


        # loss_linear
        lam = np.random.beta(1.0, 1.0)
        idx_A = torch.randperm(imgs.size(0))
        idx_B = torch.randperm(imgs.size(0))
        img_height = imgs.size(2)
        # imgs.size -> [batch_size, 3, 224, 224]
        imgs_A1 = copy.deepcopy(imgs[idx_A, :, :, :])
        imgs_B1 = copy.deepcopy(imgs[idx_B, :, :, :])
        imgs_A1 = imgs_A1[:,:, :int(lam*img_height), :]
        imgs_B1 = imgs_B1[:,:, int(lam*img_height):, :]
        imgs_M1 = torch.cat((imgs_A1, imgs_B1), 2)

        scores_A = scores[idx_A]
        scores_B = scores[idx_B]
        scores_M = model(imgs_M1).squeeze()
        
        loss_linear = nn.L1Loss()(scores_M, lam*scores_A + (1-lam)*scores_B)

        # total_loss 
        loss_train = loss_reg * 2 + loss_linear
        train_loss += loss_train.item()
        train_acc += acc_reg
        count += 1

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        scheduler.step()

        train_loader.set_postfix({'Reg loss': f'{loss_reg.item():.4f}', 'Reg Acc': f'{acc_reg:.4f}', 'Linear loss': f'{loss_linear.item():.4f}', 'Total loss': f'{loss_train.item():.4f}'})
                
    train_loss /= count
    train_acc /= count
    
    model.eval()
    val_loss = 0
    val_acc = 0
    count = 0
    val_loader = tqdm(val_loader, desc='Validation', ncols=130)
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # loss_reg
        scores = model(imgs).squeeze()
        target = torch.tensor([target_scores[label.item()] for label in labels]).to(device).float()
        loss_reg = CombinedLoss()(scores, target)
        acc_reg = torch.sum(torch.abs(scores - target) < 0.07).item() / len(scores)

        # loss_linear
        lam = np.random.beta(1.0, 1.0)
        idx_A = torch.randperm(imgs.size(0))
        idx_B = torch.randperm(imgs.size(0))
        img_height = imgs.size(2)
        imgs_A1 = copy.deepcopy(imgs[idx_A, :, :, :])
        imgs_B1 = copy.deepcopy(imgs[idx_B, :, :, :])
        imgs_A1 = imgs_A1[:,:, :int(lam*img_height), :]
        imgs_B1 = imgs_B1[:,:, int(lam*img_height):, :]
        imgs_M1 = torch.cat((imgs_A1, imgs_B1), 2)

        scores_A = scores[idx_A]
        scores_B = scores[idx_B]
        scores_M = model(imgs_M1).squeeze()

        loss_linear = nn.L1Loss()(scores_M, lam*scores_A + (1-lam)*scores_B)

        # total_loss
        loss_val = loss_reg * 2 + loss_linear
        # loss_val = loss_reg
        val_loss += loss_val.item()
        val_acc += acc_reg
        count += 1

        val_loader.set_postfix({'Reg loss': f'{loss_reg.item():.4f}', 'Reg Acc': f'{acc_reg:.4f}'})
    
    val_loss /= count
    val_acc /= count

    return train_loss, train_acc, val_loss, val_acc
   

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Input example
    cluster_number = args.cluster_num
    #os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3'

    # Graph generation mode
    graph_config = '../graph_config/'+args.graph_config  

    
    # Dataloader definition   
    # 构建POG图
    # start: [19]，发展水平最低的聚类
    # end: [1]，发展水平最高的聚类
    # partial_order: [[0, 1], [5, 0], [14, 5], [17, 14], [13, 17], [12, 13], [19, 12], [22, 19]]，POG图的backbone，22<19<12<13<17<14<5<0<1
    # cluster_unify: [[1, 2, 3, 4, 6], [0, 7], [5, 8, 9, 10], [14, 15, 16, 20], [17, 21], [13, 18], [11, 12]]，收集的是POG图中“并列”的聚类

    start, end, partial_order, cluster_unify = graph_process(graph_config)
    print(f'partial_order: {partial_order}')
    print(f'end: {end}')

    cluster_graph = generate_graph(partial_order, cluster_number)
    # cluster_path_list: [[19, 12, 13, 17, 14, 5, 0, 1]]，从起点(最低rank的19号聚类)到终点(最高rank的1号聚类)的所有路径
    cluster_path_list = cluster_graph.printPaths(start, end)
    print("Cluster_path: ", cluster_path_list)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print(f"device: {device}")
    print("loader list", cluster_unify)

    model = BackboneModel(pretrain=True, num_gpus=device_num)
    # model.initialize_weights()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)    
    target_scores = get_cluster_target_scores(partial_order, cluster_unify)
    print(f'target scores: { {k: round(v, 4) for k, v in target_scores.items()} }')

    best_loss = float('inf')
    best_acc = 0
    if args.load == False: 
        for cluster_path in cluster_path_list:
            train_loader, val_loader = make_data_loader(cluster_path, args.batch_sz)   
            print(f'length of train-loader :{len(train_loader)}')
            print(f'length of val-loader :{len(val_loader)}')

            for epoch in range(args.epochs):          
                train_loss, train_acc, val_loss, val_acc = train_and_eval(args, epoch, model, optimizer, scheduler, train_loader, val_loader, target_scores, device)

                print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if val_acc > best_acc:
                    print("state saving...")
                    state = {
                        'model': model.state_dict(),
                        'loss': val_loss,
                        'acc': val_acc,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(state, '../checkpoint/{}_best_acc'.format(args.name))
                    best_loss = val_loss
                    best_acc = val_acc
                    print(f"best loss: {best_loss:.4f}, best acc: {best_acc:.4f}")

    # 使用模型进行测试，对图片进行打分！
    print('Loading Best Model...')
    model_test = BackboneModel(pretrain=False, num_gpus=device_num)
    state_dict = torch.load('../checkpoint/{}_best_acc'.format(args.name), weights_only=True)
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
    #df2 = pd.DataFrame(columns = [ 'name', 'score'])
    cnt = 0

    with torch.no_grad():
        for batch_idx, (data, name) in enumerate(test_loader):
            print(batch_idx)
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
            df2.to_csv(args.name+'_'+args.img+'_scores.csv')
    df2.to_csv(args.name+'_'+args.img+'_scores.csv')

if __name__ == "__main__":
    args = siScore_parser()
    main(args)

