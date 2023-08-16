"rank dis + margin"
from __future__ import print_function, division
import math
import torch
import torch.nn as nn
from torchvision import transforms
from argments import parser
from comi import ResNet50
import logging
import os
import cv2
import numpy as np
import time
from preprocess_data import ImageNetData, get_data_loader
from utils import set_seed
from tqdm import tqdm



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt = '%Y-%m-%d  %H:%M:%S %a'   
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train(model, rank_dataloader, criterion, optimizer, scheduler):
    loss_dict = {}
    for i in range(args.num_classes):
        loss_dict[i] = {}

    correct = 0
    total = 0
    total_loss = 0
    len_dataloader = 0
    count = 0
    model.train()
    warmup_counter = 0
    for key in rank_dataloader:
        dataloader = rank_dataloader[key]
        len_dataloader += len(dataloader)
        
        for batch in tqdm(dataloader, desc=f"Training"):
            count += 1
            ids, images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            #print("input", images.shape, labels.shape)#(64,3,224,224)
            optimizer.zero_grad()
            outputs_m, outputs_u, features, new_features, explain_loss = model(images)
            _, predicted = torch.max(outputs_m.data, 1)
            #print("outputs", outputs.shape) #(64,10)
            label_loss = criterion(outputs_m, labels)

            for id, t, l in zip(ids, labels, label_loss):
                t = t.item()
                id = id.item()
                loss_dict[t][id] = l.item()

            label_loss = torch.mean(label_loss)
            #dis_uniform_loss = torch.mean(torch.sum(torch.softmax(outputs_u, dim=1) * torch.log_softmax(outputs_u, dim=1), dim=1))
            
            probabilities = torch.softmax(outputs_u, dim=1)
            dis_uniform_loss = torch.mean(torch.sum(-probabilities * torch.log(probabilities), dim=1))

            #dis_mse_loss = nn.MSELoss()(bert_encoding, bert_encoding_new)
            reconstruction_criterion = nn.MSELoss()
            dis_mse_loss = reconstruction_criterion(new_features, features)

            if count % 500 == 0:
                logger.info(f"label_loss  = {label_loss}")
                logger.info(f"bce_loss  = {dis_uniform_loss}")
                logger.info(f"mse_loss  = {dis_mse_loss}")
                logger.info(f"explain_loss = {explain_loss}")
            
            total_model_loss = label_loss + 0.1 * (dis_uniform_loss + dis_mse_loss + explain_loss)
            loss = total_model_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    scheduler.step()
    current_lr = scheduler.get_lr()[0]
    accuracy = 100 * correct / total

    logger.info(f'Learning Rate: {current_lr:.6f}, Train_Loss: {total_loss / len_dataloader:.4f}, Accuracy: {accuracy:.4f}%')

    return loss_dict


def evaluate(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for _, images, labels in tqdm(dataloader, desc=f"Deving"):
            images = images.to(device)
            labels = labels.to(device)
            outputs, _, _, _, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels)
            loss = torch.mean(loss)
            total_loss += loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            
    accuracy = 100 * correct / total
    '''cm = torch.zeros(args.num_classes, args.num_classes)
    for p, l in zip(all_predictions, all_labels):
        cm[p, l] += 1
    precision = cm.diag() / cm.sum(1)
    recall = cm.diag() / cm.sum(0)
    f1_score = 2 * precision * recall / (precision + recall)

    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1_score.mean().item()'''

    logger.info(f'Test Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}%')
        
    return accuracy

        
def save_pretrained(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.bin'))

def background_augmentation(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

def texture_augmentation(image):
    angle = np.random.randint(-30, 30)
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def rank_data(dataloader):
    logger.info("Start Ranking~")
    loss_dict = {}
    for i in range(args.num_classes):
        loss_dict[i] = {}
    
    model.eval()
    for batch in tqdm(dataloader, desc=f"Ranking"):
        ids, images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
    
        with torch.no_grad():
            outputs, _, _, _, _ = model(images)
            loss = loss_func(outputs, labels)
            
            
            for id, t, l in zip(ids, labels, loss):
                t = t.item()
                id = id.item()
                loss_dict[t][id] = l.item()
            
    return loss_dict


if __name__ == "__main__":
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = max([4 * torch.cuda.device_count(), 4])

    set_seed(args.seed)
    
    logger = get_logger('./logs/comi.log')#('./log/res50_dis_rank.log')
    
    logger.info(f"Args:{args}")

    train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    val_transform = transforms.Compose([
    transforms.Resize(224),
    #transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])


    image_r_label_file = "./wnids_r.txt"
    train_dir = "./imagenet/imagenet-200-train/"          
    val_dir = "./imagenet/imagenet-200-val/"
    image_r_dir = "./imagenet/in-r/imagenet-r/"
    
    train_IND = ImageNetData(args, image_r_label_file, train_dir) # [[id, image_path, label], ...]
    train_data = train_IND.data #[:1000]
    train_dataloader = get_data_loader(train_data, args.batch_size*3, transform=train_transform, shuffle=True)
    logger.info(f"len_train_data: {len(train_data)}")   
    logger.info(f"num_batches_train: {len(train_dataloader)}") 

    val_IND = ImageNetData(args, image_r_label_file, val_dir) # [[id, image_path, label], ...]
    val_data = val_IND.data#[:1000]
    val_dataloader = get_data_loader(val_data, batch_size=1, transform=val_transform, shuffle=False)
    logger.info(f"len_val_data: {len(val_data)}")   
    logger.info(f"num_batches_val: {len(val_dataloader)}") 

    image_r_IND = ImageNetData(args, image_r_label_file, image_r_dir) # [[id, image_path, label], ...]
    image_r_data = image_r_IND.data
    image_r_dataloader = get_data_loader(image_r_data, batch_size=1, transform=val_transform, shuffle=False)
    logger.info(f"len_image_r_data: {len(image_r_data)}")   
    logger.info(f"num_batches_image_r: {len(image_r_dataloader)}") 

    
    model = ResNet50(args, device).to(device)
    #summary.summary(model, input_size=(3,224,224), device="cpu")

    criterion = nn.CrossEntropyLoss(reduction='none')
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#0.001  0.005 0.01 0.0001 0.0005 0.00005
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1)
    
    scheduler_setting = "warmuplinear"
    
    
    loss_func = nn.CrossEntropyLoss(reduction='none') 

    #######CoHa
    logger.info(f"Rank radio: {args.rank_percent}")
    # loss_dict = {n:0 for n in range(1, len(train_data)+1)}
    loss_dict = rank_data(train_dataloader)
    
    total_len = 0
    for i in loss_dict:
        total_len += len(loss_dict[i])
        logger.info(f"len-{i} = {len(loss_dict[i])}")
        
    logger.info(f"total-len = {total_len}")
    
    # reverse=True #from high to low
    sort_loss_dict = []
    for i in range(args.num_classes):
        sort_loss_dict.append(sorted(loss_dict[i].items(), key=lambda x: x[1], reverse=True))
    

    rank_train_data = {}
    for i in range(2):
        rank_train_data[i] = []

    for i in range(len(sort_loss_dict)):
        length_sort_loss_dict = len(sort_loss_dict[i])
        index1 = int(length_sort_loss_dict * args.rank_percent) # 0.1
        for j in range(length_sort_loss_dict):
            item = sort_loss_dict[i][j]
            if j < index1: # high loss
                rank_train_data[0].append(train_data[item[0]])
            else:
                rank_train_data[1].append(train_data[item[0]])
                # logger.info("rank not enough！")
    
    
    # del train_dataloader
    torch.cuda.empty_cache()

    rank_train_dataloader = {}
    for i in range(2):
        rank_train_dataloader[i] = get_data_loader(rank_train_data[i], args.batch_size, transform=train_transform, shuffle=True)

    test_flag = False
    best_acc = 0.
    best_epoch = 0

    best_acc1 = 0.
    best_epoch1 = 0
    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

    for epoch in range(1, args.epoch_num + 1):
        logger.info(f"Epoch {epoch}:")
        loss_dict = train(model, rank_train_dataloader, criterion, optimizer, scheduler)
        logger.info("VAL:")
        acc = evaluate(model, val_dataloader)
        logger.info("R:")
        acc_r = evaluate(model, image_r_dataloader)

        #####
        '''sort_loss_dict = []
        for i in range(args.num_classes):
            sort_loss_dict.append(sorted(loss_dict[i].items(), key=lambda x: x[1], reverse=True))

        hl_train_data, ll_train_data = [], []  
        for i in range(len(sort_loss_dict)):
            h_index = int(len(sort_loss_dict[i]) * args.rank_percent)
            for j in range(len(sort_loss_dict[i])):
                item = sort_loss_dict[i][j]
                if j < h_index: # high loss
                    hl_train_data.append(train_data[item[0]])
                else: # low loss
                    ll_train_data.append(train_data[item[0]])
        
        hl_train_dataloader = get_data_loader(hl_train_data, args.batch_size, transform=train_transform, shuffle=True)
        ll_train_dataloader = get_data_loader(ll_train_data, args.batch_size, transform=train_transform, shuffle=True)'''



        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            
            checkpoints_dirname = f"outputs_rank/res50_rank_v_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname)
        
        if acc_r > best_acc1:
            best_acc1 = acc_r
            best_epoch1 = epoch
            
            checkpoints_dirname = f"outputs_rank/res50_rank_r_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname)
               
    logger.info(f"Val: Best Acc = {best_acc}, epoch {best_epoch}")     
    logger.info(f"R: Best Acc = {best_acc1}, epoch {best_epoch1}")  

