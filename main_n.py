import torch
import json
import random
import numpy as np
import math
import time
from tqdm import tqdm
import os 
import logging
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from compute_ce import get_CE_metrics
from model_n import MyBertModel
from argments_nlp import parser
###from argments import parser
from utils import set_seed
from preprocess import MNLIData
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt = '%Y-%m-%d  %H:%M:%S %a'    #attention
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


def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler
        """
        scheduler = scheduler.lower()
        if scheduler=='constantlr':
            return get_constant_schedule(optimizer)
        elif scheduler=='warmupconstant':
            return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler=='warmuplinear':
            return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler=='warmupcosine':
            return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler=='warmupcosinewithhardrestarts':
            return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, predicted, target):
        # Normalize the vectors to unit length
        predicted_normalized = F.normalize(predicted, p=2, dim=1)
        target_normalized = F.normalize(target, p=2, dim=1)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(predicted_normalized, target_normalized, dim=1)

        # Use cosine similarity as the loss (minimize 1 - cosine similarity)
        loss = 1 - cosine_sim.mean()

        return loss


def logit_reg(dataloader, epoch):
    for batch in tqdm(dataloader, desc=f"Eval Epoch {epoch}"):
        _, input_texts, input_tokens, inputs, targets  = batch
      
        inputs = inputs.to(device)
        targets = targets.to(device)
      
        exp_loss, bert_output, bert_output_u, seq_encoding, seq_encoding_new = model(input_texts, input_tokens, inputs)
        
        labels = targets.cpu().detach().numpy().reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        one_hot_labels = encoder.fit_transform(labels)   
           
        probabilities = torch.softmax(bert_output, dim=1)
        
        platt_scalers = []
        platt_losses = 0.0
        for i in range(one_hot_labels.shape[1]):
            platt_scaler = LogisticRegression().fit(bert_output.cpu().detach().numpy(), one_hot_labels[:, i])
            predictions = platt_scaler.predict_proba(bert_output.cpu().detach().numpy())
            loss = log_loss(one_hot_labels[:, i], predictions)
            platt_scalers.append(platt_scaler)
            platt_losses += loss
            
        #platt_scaler = LogisticRegression().fit(bert_output.cpu().detach().numpy().reshape(-1, 1), one_hot_labels.reshape(-1, 1))
    return platt_scalers



def train(rank_dataloader):
    loss_dict = {
        0 : {},
        1 : {},
        2 : {}
    }
    #model.train()
    model.train()
    # total loss in epoch
    total_loss = 0
    # tqdm
    y_true = []
    y_pred = []
    y_true_one_hot = []
    labels_oneh = []
    preds = []
    len_dataloader = 0

    start = time.time()
    count = 0
    count_patience_exp = 0
    '''output_file_result = open(f"./result1/{epoch}_train_output.txt", "a")
    output_explain_result = open(f"./result1/{epoch}_train_explain.json", "a")'''
    for key in rank_dataloader:
        dataloader = rank_dataloader[key]
        len_dataloader += len(dataloader)
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
            count += 1
            
            ids, input_texts, input_tokens, inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            #
            optimizer.zero_grad()

            #
            exp_loss, bert_output_m, bert_output_u, seq_encoding, seq_encoding_new = model(input_texts, input_tokens, inputs, flag_explain=True)

            probabilities_m = torch.softmax(bert_output_m, dim=1)

            ce_loss = loss_func(bert_output_m, targets)
            
            # for id, t, l in zip(ids, targets, ce_loss):
            #     t = t.item()
            #     loss_dict[t][id] = l.item()
            ce_loss_mean = torch.mean(ce_loss)

            batch_size = bert_output_u.size(0)

            loss_fn = nn.KLDivLoss(reduction='batchmean')
            uniform_distribution = torch.full((batch_size, args.num_classes), 1.0 / args.num_classes)
            uniform_distribution = uniform_distribution.to(bert_output_u.device).float()
            u_logits = F.softmax(bert_output_u, dim=1)
            dis_uniform_loss = loss_fn(torch.log(u_logits+1e-10), uniform_distribution)

            reconstruction_criterion = nn.MSELoss()
            dis_mse_loss = reconstruction_criterion(seq_encoding, seq_encoding_new)
            
           
            if count % 1000 == 0:
                logger.info(f"exp_loss = {exp_loss}")
                logger.info(f"ce_loss_mean  = {ce_loss_mean}")
                logger.info(f"dis_uni_loss  = {dis_uniform_loss}")
                logger.info(f"dis_mse_loss  = {dis_mse_loss}")

            
            loss = ce_loss_mean + args.loss_weight_dis * dis_mse_loss + args.loss_weight_exp * exp_loss

            # 
            loss.backward()

            # 
            optimizer.step()
            
            # update training rate
            scheduler.step()

            y_one_hot = torch.nn.functional.one_hot(targets, num_classes=3)
            y_true_one_hot.extend(y_one_hot)
            y_pred.extend(probabilities_m.argmax(dim=1).cpu().numpy())#bert_output_m
            y_true.extend(targets.cpu().numpy())

            # 
            total_loss += loss.item()

            label_oneh = torch.nn.functional.one_hot(targets, num_classes=3)
            label_oneh = label_oneh.cpu().detach().numpy()
            labels_oneh.extend(label_oneh)

            pred = probabilities_m.cpu().detach().numpy()
            preds.extend(pred)

            '''try:
                for id, explain in zip(ids, explain_list):
                    output_explain_result.write(str(id)+'\t')
                    json.dump(explain, output_explain_result)
                    output_explain_result.write('\n')

                for id, loss, prob, y_one_hot, pre, y in zip(ids, ce_loss, probabilities_m, y_true_one_hot, y_pred, y_true):
                    line = f"{id}\t{loss.cpu().detach().numpy()}\t{prob.cpu().detach().numpy()}\t{y_one_hot.cpu().detach().numpy()}\t\t{pre}\t{y}\t{True if pre==y else False}\n"  # 使用制表符分隔两列数据，可以根据需要修改分隔符
                    output_file_result.write(line)
        
            except:
                print(ids)
                print(explain_list)
            '''

    end = time.time()
    mytime = end - start
    
    f1 = f1_score(y_true, y_pred, average='macro')
    pre = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    preds = np.array(preds).flatten()
    labels_oneh = np.array(labels_oneh).flatten()
    ece, mce = get_CE_metrics(preds, labels_oneh)
    
    logger.info(f"ece = {ece:.4f}, mce= {mce:.4f}")
    logger.info(f"Train: Loss = {total_loss / ( len_dataloader):.4f}, Accuracy = {acc:.4f}, F1 = {f1:.4f}, Precision = {pre:.4f}, Recall = {recall:.4f}, Time: {mytime//60:.2f}m {mytime%60:.2f}s")
    
    return loss_dict

def evaluate(dataloader, platt_scalers, flag_dataset, flag_hans=False):
    #Evaluation
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    labels_oneh = []
    preds = []
    start = time.time()
    '''output_file_m = open("./result1/m_output_label.txt", "a")
    output_file_mm = open("./result1/mm_output_label.txt", "a")
    output_file_hans = open("./result1/hans_output_label.txt", "a")'''
    for batch in tqdm(dataloader, desc=f"deving"):
        _, input_texts, input_tokens, inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            _, bert_output, _, _, _ = model(input_texts, input_tokens, inputs)
            new_probabilities = torch.softmax(bert_output, dim=1)
            
            loss = loss_func(bert_output, targets)
            
            loss = torch.mean(loss)
            
            total_loss += loss.item()

            if flag_hans: 
                new_probabilities = []
                for platt_scaler in platt_scalers:
                    probab = platt_scaler.predict_proba(bert_output.cpu().detach().numpy())
                    # which is the probability that the sample belongs to class 1
                    probab = probab[:, 1]
                    new_probabilities.append(probab)
                new_probabilities = np.array(new_probabilities).squeeze(axis=1)
                new_probabilities = torch.tensor(new_probabilities).unsqueeze(0) 
                         
                max_values, _ = new_probabilities[:, [0, 2]].max(dim=1)
                max_values = max_values.to(new_probabilities.device)
                new_probabilities[:, 0] = max_values
                new_probabilities = new_probabilities[:, :2]
            
            

            y_pred.extend(new_probabilities.argmax(dim=1).cpu().numpy()) #bert_output
            y_true.extend(targets.cpu().numpy())
            
            
            pred = new_probabilities.cpu().detach().numpy()
            preds.extend(pred)
            
            if flag_hans:
                label_oneh = torch.nn.functional.one_hot(targets, num_classes=2)
            else:
                label_oneh = torch.nn.functional.one_hot(targets, num_classes=3)

            label_oneh = label_oneh.cpu().detach().numpy()
            labels_oneh.extend(label_oneh)

            '''
            file_path = "output.txt"
            if flag_dataset == "dev-m":
                for pre, gt in zip(probabilities.argmax(dim=1), targets):
                    line = f"{pre}\t{gt}\n"  
                    output_file_m.write(line)
            elif flag_dataset == "dev-mm":
                for pre, gt in zip(probabilities.argmax(dim=1), targets):
                    line = f"{pre}\t{gt}\n"  
                    output_file_mm.write(line)
            elif flag_dataset == "hans":
                for pre, gt in zip(probabilities.argmax(dim=1), targets):
                    line = f"{pre}\t{gt}\n"  
                    output_file_hans.write(line)'''
    
    # print(f"Acc: {acc / len(dev_dataloader):.2f}")
    end = time.time()
    mytime = end - start

    '''if flag_hans:
        y_pred = [y if y == 1 else 0 for y in y_pred]'''

    f1 = f1_score(y_true, y_pred, average='macro')
    pre = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    preds = np.array(preds).flatten()
    labels_oneh = np.array(labels_oneh).flatten()
    ece, mce = get_CE_metrics(preds, labels_oneh)

    logger.info(f"ece = {ece:.4f}, mce= {mce:.4f}")
    
    logger.info(f"{flag_dataset} : Loss = {total_loss/len(dataloader):.4f}, Accuracy = {acc:.4f}, F1 = {f1:.4f}, Precision = {pre:.4f}, Recall = {recall:.4f}, Time: {mytime//60:.2f}m {mytime%60:.2f}s")
    
    return acc, f1




def save_pretrained(model, path):#weights
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'model_weights.bin'))


def rank_data(dataloader):
    logger.info("Start Ranking~")
    loss_dict = {
        0 : {},
        1 : {},
        2 : {}
    }
    
    model.eval()
    for batch in tqdm(dataloader, desc=f"Ranking"):
        # tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
    
        ids, input_texts, input_tokens, inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            _, bert_output, _, _, _, _ = model(input_texts, input_tokens, inputs, flag_explain=False)
            #model(inputs)
            
            loss = loss_func(bert_output, targets)
            
            # logger.info(f"pred = {bert_output, bert_output.shape}")
            # logger.info(f"target = {targets, targets.shape}")
            # logger.info(f"Loss = {loss, loss.shape}")
            # logger.info(f"ID = {ids}")
            
            for id, t, l in zip(ids, targets, loss):
                t = t.item()
                loss_dict[t][id] = l.item()
    return loss_dict


def rank_data(dataloader):
    logger.info("Start Ranking~")
    loss_dict = {
        0 : {},
        1 : {},
        2 : {}
    }
    
    model.eval()
    for batch in tqdm(dataloader, desc=f"Ranking"):
        # tqdm(train_dataloader, desc=f"Training Epoch {epoch}") 会自动执行DataLoader的工作流程，

        ids, input_texts, input_tokens, inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            # model.forward(inputs)
            _, bert_output, _, _, _ = model(input_texts, input_tokens, inputs)
            #model(inputs)
            
            loss = loss_func(bert_output, targets)
            
            # logger.info(f"pred = {bert_output, bert_output.shape}")
            # logger.info(f"target = {targets, targets.shape}")
            # logger.info(f"Loss = {loss, loss.shape}")
            # logger.info(f"ID = {ids}")f
            
            for id, t, l in zip(ids, targets, loss):
                t = t.item()
                loss_dict[t][id] = l.item()
    return loss_dict
    

if __name__ == "__main__":
    args = parser()
    torch.cuda.set_device(args.gpu)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = max([4 * torch.cuda.device_count(), 4])

    set_seed(args.seed)
    
    logger = get_logger(f'logs/{args.log_name}.log')####
    
    logger.info(f"Args:{args}")

    ###data processing
    ###Other data sets can be replaced here
    mnli_dataset = MNLIData(args)
    train_data = mnli_dataset.train_data
    dev_data = mnli_dataset.dev_data
    mis_dev_data = mnli_dataset.mis_dev_data
    hans_eval_data = mnli_dataset.hans_eval_data

    
    scheduler_setting = "warmuplinear"
    total_steps = math.ceil(args.epoch_num * len(train_data)*1./args.batch_size)
    warmup_steps = int(total_steps * args.warmup_percent)


    num_train = int(len(train_data) * args.train_percent)
    train_dataset = mnli_dataset.get_dataset(train_data[:num_train])
    eval_dataset = mnli_dataset.get_dataset(train_data[num_train:])
    
    
    dev_dataset = mnli_dataset.get_dataset(dev_data)
    mis_dev_dataset = mnli_dataset.get_dataset(mis_dev_data)
    hans_eval_dataset = mnli_dataset.get_dataset(hans_eval_data)

    
    train_dataloader = mnli_dataset.get_dataloader(
                                        train_dataset,
                                        batch_size=args.batch_size*4,
                                        shuffle=True,
                                    )
    
    eval_dataloader = mnli_dataset.get_dataloader(
                                    eval_dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    )
    
    dev_dataloader = mnli_dataset.get_dataloader(
                                    dev_dataset,
                                    batch_size=1,
                                )
    mis_dev_dataloader = mnli_dataset.get_dataloader(
                                mis_dev_dataset,
                                batch_size=1,
                            )
    
    hans_eval_dataloader = mnli_dataset.get_dataloader(
                                hans_eval_dataset,
                                batch_size=1,
                            )
    
    model = MyBertModel(args, device)
    
    model.to(device)
    
    
    # layer freezing
    no_grad_param_names = ["embeddings"] + [
        "layer.{}.".format(i) for i in range(args.freeze_layer)
    ]
    logger.info("The frozen parameters are:")
    for name, param in model.named_parameters():
        if any(no_grad_pn in name for no_grad_pn in no_grad_param_names):
            param.requires_grad = False
            logger.info("  {}".format(name))
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in param_optimizer
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, args.lr, weight_decay=0.001)  #使用AdamW优化器
    scheduler = get_scheduler(optimizer, scheduler_setting, warmup_steps=warmup_steps, t_total=total_steps)
    
    loss_func = nn.CrossEntropyLoss(reduction='none')  # 使用crossentropy作为损失函数
    
    ###### ranking
    ###### accoding to losses according to bert
    logger.info(f"Rank radio: {args.rank_percent}")
    # loss_dict = {n:0 for n in range(1, len(train_data)+1)}
    loss_dict = rank_data(train_dataloader)
    
    total_len = 0
    for i in loss_dict:
        total_len += len(loss_dict[i])
        logger.info(f"len-{i} = {len(loss_dict[i])}")
        
    logger.info(f"total-len = {total_len}")
    
    # reverse=True Sort loss from highest to lowest
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
                # logger.info("Data rank is not complete!")
    
    
    # del train_dataloader
    torch.cuda.empty_cache()

    rank_train_dataloader = {}
    for i in range(2):
        tmp_dataset = mnli_dataset.get_dataset(rank_train_data[i])
        rank_train_dataloader[i] = mnli_dataset.get_dataloader(tmp_dataset, args.batch_size, shuffle=True)
        

    rank_train_dataloader = {}
    for i in range(2):
        tmp_dataset = mnli_dataset.get_dataset(rank_train_data[i])
        rank_train_dataloader[i] = mnli_dataset.get_dataloader(tmp_dataset, args.batch_size, shuffle=True)
    
    
    logger.info('Start Training!')
    
    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())
    best_f1 = 0.0
    best_acc = 0.0
    best_epoch1 = 0
    best_epoch2 = 0
    
    best_mis_f1 = 0.0
    best_mis_acc = 0.0
    best_mis_epoch1 = 0
    best_mis_epoch2 = 0
    
    best_hans_f1 = 0.0
    best_hans_acc = 0.0
    best_hans_epoch1 = 0
    best_hans_epoch2 = 0

    for epoch in range(1, args.epoch_num + 1):
        logger.info(f"Epoch {epoch}:")
        loss_dict = train(rank_train_dataloader)
        platt = logit_reg(eval_dataloader, epoch)
                
        acc, f1 = evaluate(dev_dataloader, platt, "dev-m")
        mis_acc, mis_f1 = evaluate(mis_dev_dataloader, platt, "dev-mm")
        hans_acc, hans_f1 = evaluate(hans_eval_dataloader, platt, "hans", flag_hans=True)
        logger.info('\n')

        
        if f1 > best_f1:
            best_f1 = f1
            best_epoch2 = epoch
        
        if acc > best_acc:
            best_acc = acc
            best_epoch1 = epoch
            
            checkpoints_dirname = f"model_outputs/{args.log_name}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname+ 'dev/')
            
        if mis_f1 > best_mis_f1:
            best_mis_f1 = mis_f1
            best_mis_epoch2 = epoch
        
        if mis_acc > best_mis_acc:
            best_mis_acc = mis_acc
            best_mis_epoch1 = epoch
            
            checkpoints_dirname = f"model_outputs/{args.log_name}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname + 'mismatch/')
            
        if hans_f1 > best_hans_f1:
            best_hans_f1 = hans_f1
            best_hans_epoch2 = epoch
        
        if hans_acc > best_hans_acc:
            best_hans_acc = hans_acc
            best_hans_epoch1 = epoch
            
            checkpoints_dirname = f"model_outputs/{args.log_name}_" + timestamp + '/'
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model,
                            checkpoints_dirname + 'hans/')
            
    
    ####recoding the best results
    logger.info("Match:")    
    logger.info(f"Best Acc = {best_acc}, epoch {best_epoch1}")    
    logger.info(f"Best F1  = {best_f1}, epoch {best_epoch2}")
    
    logger.info('\n')
    logger.info(f"Mismatch:")
    logger.info(f"Best Acc = {best_mis_acc}, epoch {best_mis_epoch1}")    
    logger.info(f"Best F1  = {best_mis_f1}, epoch {best_mis_epoch2}")

    logger.info('\n')
    logger.info(f"Hans:")
    logger.info(f"Best Acc = {best_hans_acc}, epoch {best_hans_epoch1}")    
    logger.info(f"Best F1  = {best_hans_f1}, epoch {best_hans_epoch2}")




    
