import torch
from torch.utils.data import Dataset, DataLoader
import os
from transformers import BertTokenizerFast
import logging
import csv
import sys

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__file__)


class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # You can define it yourself，Dataloader get data from __getitem__(self, index)
        
        return self.dataset[index]
    

class MNLIData(Dataset):
    def __init__(self, args):
        self.label_dict ={'contradiction': 0, 'entailment': 1, 'neutral': 2, 'non-entailment': 0}  

        self.args = args
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        
        self.train_data_path = os.path.join(args.data_path, 'train.tsv')
        self.dev_data_path = os.path.join(args.data_path, 'dev_matched.tsv')
        self.mis_dev_data_path = os.path.join(args.data_path, 'dev_mismatched.tsv')
        # self.test_data_path = os.path.join(self.args.data_path, 'test_matched.tsv')
        # self.mis_test_data_path = os.path.join(self.args.data_path, 'test_mismatched.tsv')
        self.hans_eval_data_path = os.path.join(args.data_path, 'heuristics_evaluation_set.tsv')

        self.tokenizer = BertTokenizerFast.from_pretrained(args.bert_path, do_lower_case=True)
        
        self.train_data = None
        self.dev_data = None
        self.mis_dev_data = None
        self.train_text = []
        self.train_label = []
        # self.test_data = None
        # self.mis_test_data = None

        self.hans_eval_data = None
        self.init_data()

    def init_data(self):
        self.train_data = self.load_data(self.train_data_path, True)
        self.dev_data = self.load_data(self.dev_data_path)
        self.mis_dev_data = self.load_data(self.mis_dev_data_path)
        # self.test_data = self.load_data(self.test_data_path)
        # self.mis_test_data = self.load_data(self.mis_test_data_path)
        self.hans_eval_data = self.load_hans(self.hans_eval_data_path)

    def load_data(self, path, flag=False):
        logger.info('Loading data....{}'.format(path))
        result = []
        with open(path, 'r') as f:
            f.readline()
            lines = f.readlines()
            tsvreader = csv.reader(f, delimiter='\t')
            '''column = next(tsvreader)
            s1 = column.index('sentence1')
            s2 = column.index('sentence2')
            y = column.index('gold_label')'''
            for line in lines:
                line = line.split("\t")
                id = int(line[0])
                text = line[8] + '[SEP]' + line[9]
                label = self.label_dict[line[-1].rstrip()]
                if flag == True:
                    self.train_text.append(text)
                    self.train_label.append(label)
                    result.append([id, text, label])
                else:
                    result.append([id, text, label])
        logger.info(f"len = {len(result)}")
        return result  

    def load_hans(self, path):
        logger.info('Loading data....{}'.format(path))
        result = []
        id = 0
        with open(path, 'r') as f:
            
            f.readline()
            lines = f.readlines()
            tsvreader = csv.reader(f, delimiter='\t')
            '''column = next(tsvreader)
            s1 = column.index('sentence1')
            s2 = column.index('sentence2')
            y = column.index('gold_label')'''
            for line in lines:
                id += 1
                line = line.split("\t")
                text = line[5] + '[SEP]' + line[6]
                label = self.label_dict[line[0]]
                type = line[8]
                
                result.append([id, text, label])
        logger.info(f"hans len = {len(result)}")
        return result 
    
    def get_dataset(self, data):
        return BertDataset(data)
    
    def collate_fn(self, examples):
        X, Y = [], []
        ID = []
        X_tokens = []
        # print("~~~~~~~~", examples)
        for id, x, y in examples:
            ID.append(id)
            X.append(x)
            X_tokens.append(self.tokenizer.tokenize(x, max_length=self.max_length, truncation=True))
            Y.append(int(y))
        
        X_tensor = self.tokenizer(X,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length)
        Y = torch.tensor(Y)
        return ID, X, X_tokens, X_tensor, Y
    
    
         
    def get_dataloader(self, dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True):
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )
        
        return dataloader
    

class QQPData(Dataset):
    def __init__(self, args):
        
        self.args = args
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        
        self.train_data_path = os.path.join(args.qqp_data_path, 'train.tsv')
        self.dev_data_path = os.path.join(args.qqp_data_path, 'dev.tsv')
        # self.test_data_path = os.path.join(self.args.data_path, 'test_matched.tsv')
        # self.mis_test_data_path = os.path.join(self.args.data_path, 'test_mismatched.tsv')
        self.paws_eval_data_path = os.path.join(args.qqp_data_path, 'paws_validation.tsv')

        self.tokenizer = BertTokenizerFast.from_pretrained(args.bert_path, do_lower_case=True)
        
        self.train_data = None
        self.dev_data = None
        self.train_text = []
        self.train_label = []
        # self.test_data = None
        # self.mis_test_data = None

        self.paws_eval_data = None
        self.init_data()

    def init_data(self):
        self.train_data = self.load_data(self.train_data_path, flag=True)
        self.dev_data = self.load_data(self.dev_data_path)
        # self.test_data = self.load_data(self.test_data_path)
        # self.mis_test_data = self.load_data(self.mis_test_data_path)
        self.paws_eval_data = self.load_data(self.paws_eval_data_path, flag_paws=True)

    def load_data(self, path, flag=False, flag_paws=False):
        logger.info('Loading data....{}'.format(path))
        result = []
        id = 0
        with open(path, 'r') as f:
            if flag_paws == True:
                tsvreader = csv.reader(f, delimiter=',')
                column = next(tsvreader)
                s1 = column.index('sentence1')
                s2 = column.index('sentence2')
                y = column.index('label')
            else:
                tsvreader = csv.reader(f, delimiter='\t')
                column = next(tsvreader)
                s1 = column.index('question1')
                s2 = column.index('question2')
                y = column.index('is_duplicate')
            
            for line in tsvreader:
                try:
                    text = line[s1] + '[SEP]' + line[s2]
                    label = line[y]
                    result.append([id, text, label])
                    id += 1
                    if flag == True:
                        self.train_text.append(text)
                        self.train_label.append(label)
                except:
                    print(id)
            
        logger.info(f"len = {len(result)}")
        return result   
    
    def get_dataset(self, data):
        return BertDataset(data)
    
    def collate_fn(self, examples):
        X, Y = [], []
        ID = []
        X_tokens = []
        # print("~~~~~~~~", examples)
        for id, x, y in examples:
            ID.append(id)
            X.append(x)
            X_tokens.append(self.tokenizer.tokenize(x, max_length=self.max_length, truncation=True))
            Y.append(int(y))
        X_tensor = self.tokenizer(X,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length)
        Y = torch.tensor(Y)
        return ID, X, X_tokens, X_tensor, Y
         
    def get_dataloader(self, dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True):
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )
        return dataloader
