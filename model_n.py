from transformers import BertModel, BertTokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import numpy as np
import logging
explainer = LimeTextExplainer()

logger = logging.getLogger(__file__)

    
class MyBertModel(nn.Module):
    def __init__(self, args, device):
        super(MyBertModel, self).__init__()
        self.device = device
        self.max_length = args.max_length
        self.encoder_type = args.encoder_type
        self.bert = BertModel.from_pretrained(args.bert_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, args.num_classes)
        self.exp_project = nn.Linear(768, 1)
        self.dis_exp_project = nn.Linear(args.num_classes, 1)
        self.correlated_branch = nn.Sequential(
            #nn.Linear(768, 256),
            #nn.ReLU(),
            nn.Linear(768, args.num_classes),
            nn.ReLU()
        )
        self.uncorrelated_branch = nn.Sequential(
            #nn.Linear(768, 256),
            #nn.ReLU(),
            nn.Linear(768, args.num_classes),
            nn.ReLU()
        )
        self.reconstruction_branch = nn.Sequential(
            nn.Linear(args.num_classes * 2, 768),
            nn.ReLU()
            #nn.ReLU(),
            #nn.Linear(256, 768)
        )
        

    def forward(self, input_texts, input_tokens, inputs, flag_explain=False):
        #input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        #bert_output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        bert_output = self.get_bert(inputs)
        sequence_output = bert_output.last_hidden_state # (batch_size, max_len, hidden_size)
        output_m, output_u, seq_encoding, seq_encoding_new = self.disentangle_states(sequence_output)
        output_bert_m = torch.avg_pool1d(output_m.transpose(1, 2), kernel_size=output_m.size(1) ).squeeze(-1)
        output_bert_u = torch.avg_pool1d(output_u.transpose(1, 2), kernel_size=output_m.size(1) ).squeeze(-1)# (batch_size, hidden_size)
        if flag_explain: # 
            exp_sequence_output = self.dis_exp_project(output_m)#sequence_output ##
            explains_list = self.lime_explain(input_texts, input_tokens) # 
            # logger.info(f"sta: {self.stastics_explanation(explains_list)}")
            avg_weight = self.stastics_explanation(explains_list)
            avg_weight = avg_weight if avg_weight > 0 else 0.  #
            exp_loss = self.calc_explain_loss(exp_sequence_output, 
                                            input_tokens,
                                            explains_list, 
                                            avg_weight)
            print(f"exp_loss:{exp_loss}")
        else:
            exp_loss = 0
            explains_list = None
        
        return exp_loss, output_bert_m, output_bert_u, seq_encoding, seq_encoding_new   #self.get_output(bert_output)
    
    def get_bert(self, inputs):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        bert_output = self.bert(input_ids, attention_mask, output_hidden_states=True)
        return bert_output
    
    def predict_fn(self, text):
        inputs = self.tokenizer(text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length)
        inputs = inputs.to(self.device)
        bert_output = self.get_bert(inputs)
        ex_outputs = self.get_output(bert_output)
        del inputs
        return torch.softmax(ex_outputs, dim=1).cpu().detach().numpy()
    

    def lime_explain(self, input_texts, input_tokens):
        weights_list = []
        exp_list = []
        for text, tokens in zip(input_texts, input_tokens):
            feature_len = max(len(set(tokens)) - 1, 10)
            # generate explain
            exp = explainer.explain_instance(text, self.predict_fn, num_features=feature_len, num_samples=5)
            exp_list.append(exp)
        
        for exp in exp_list:
            weights_dict = {}
            for feature, weight in list(exp.as_list()):
                feature_tokens = self.tokenizer.tokenize(feature, max_length=self.max_length, truncation=True)
                for feature_token in feature_tokens:
                    weights_dict[feature_token] = weight
            weights_list.append(weights_dict)
       
        return weights_list
    

    
    def stastics_explanation(self, explains_list):
        weights_list = [w for explain in explains_list for _, w in explain.items()]
        weights_list = np.array(weights_list)
        return np.mean(weights_list)
    
    def calc_explain_loss(self, exp_sequence_output, input_tokens, explains_list, avg_weight):
        explain_loss = 0
        exp_sequence_output = exp_sequence_output.clone().squeeze(-1)
        exp_mask = torch.zeros_like(exp_sequence_output).bool()    
        for i in range(len(explains_list)):# list里有 batch_size 个dict，dict的key是token，value是weight
            explain_dict = explains_list[i]
            tokens = input_tokens[i]
            for j in range(len(tokens)):
                token = tokens[j]
                if token in explain_dict.keys():
                    if explain_dict[token] > avg_weight:####>
                        exp_mask[i, j] = True
        
        for i in range(exp_sequence_output.size(0)):            
            tmp_exp_seq_output = torch.masked_select(exp_sequence_output[i, :], exp_mask[i, :]) 
            tmp_exp_seq_output = F.normalize(tmp_exp_seq_output, p=2, dim=-1)
            percentile_index = int(0.5 * len(tmp_exp_seq_output))
            sorted_tensor, _ = torch.sort(tmp_exp_seq_output, descending=True)#
            if len(sorted_tensor) != 0:
                margin = sorted_tensor[percentile_index] #
            else:
                margin = 0

            #margin = torch.median(tmp_exp_seq_output) # 
            tmp_exp_seq_output = torch.where(tmp_exp_seq_output > margin, tmp_exp_seq_output - margin, margin - tmp_exp_seq_output)  
            tmp_exp_loss = tmp_exp_seq_output.sum() / (tmp_exp_seq_output.size(0) + 1e-10)
            explain_loss += tmp_exp_loss
        
        return explain_loss
    

    def min_max_normalize(self, matrix):
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)
        print(f"{min_val}, {max_val}")
        normalized_matrix = (matrix - min_val) / (max_val - min_val + 1e-10)
        return normalized_matrix

    def output_type(self, bert_output, encoder_type):
        if encoder_type == 'fist-last-avg':
        
            first = bert_output.hidden_states[1]   # 
            last = bert_output.hidden_states[-1]
            seq_length = first.size(1)   # seq_len

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=2).squeeze(-1)
            return final_encoding

        if encoder_type == 'last-avg':
            sequence_output = bert_output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if encoder_type == "cls":
            sequence_output = bert_output.last_hidden_state
            cls = sequence_output[:, 0]  # [b, hidden_size]
            return cls

        if encoder_type == "pooler":
            pooler_output = bert_output.pooler_output  # [b,d]
            return pooler_output
        
    def get_output(self, bert_output):
        output = self.output_type(bert_output, self.encoder_type)
        output = self.dropout(output)
        output = self.classifier(output)
        return output
    
    def disentangle_states(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        output_m = self.correlated_branch(hidden_states)#(b,s,c)
        output_u = self.uncorrelated_branch(hidden_states)#(b,s,c)
        reconstructed_output = self.reconstruction_branch(torch.cat((output_m, output_u), dim=2))
        return output_m, output_u, hidden_states, reconstructed_output
