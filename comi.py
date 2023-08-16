import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import logging
logger = logging.getLogger(__file__)


class ResNet50(nn.Module):
    def __init__(self, args, device):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.device = device
        self.num_features = self.resnet.fc.in_features #2048
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.label_predictor = nn.Linear(args.info_dim, args.num_classes)
        
        self.relevant_info_capturer = nn.Linear(self.num_features, args.info_dim)
        self.irrelevant_info_capturer = nn.Linear(self.num_features, args.info_dim)
        self.reconstruction_capturer = nn.Linear(args.info_dim*2, self.num_features)
        self.label_predictor = nn.Linear(args.info_dim, args.num_classes)
        self.texture_capturer = nn.Linear(self.num_features, args.num_classes)
        self.bg_capturer = nn.Linear(self.num_features, args.num_classes)
        self.correlated_branch = nn.Sequential(
            nn.Linear(self.num_features, args.info_dim),
            nn.ReLU(),
            nn.Linear(args.info_dim, args.num_classes)
        )
        self.uncorrelated_branch = nn.Sequential(
            nn.Linear(self.num_features, args.info_dim),
            nn.ReLU(),
            nn.Linear(args.info_dim, args.num_classes)
        )
        self.reconstruction_branch = nn.Sequential(
            nn.Linear(args.num_classes * 2, args.info_dim),
            nn.ReLU(),
            nn.Linear(args.info_dim, self.num_features)
        )
        self.correlated_fc = nn.Linear(args.info_dim, args.num_classes)
        self.uncorrelated_fc = nn.Linear(args.info_dim, args.num_classes)
        self.nn_sig = nn.Sigmoid()

        

    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        outputs_m, outputs_u, new_features = self.disentangle_feature(features)

    
        explain_loss = self.calculate_margin(outputs_m)
       
        #texture_outputs = self.texture_capturer(texture_features) 
        #bg_outputs = self.bg_capturer(bg_features) 
    
        return outputs_m, outputs_u, features, new_features, explain_loss
    

    def disentangle_feature(self, features):
        outputs_m = self.correlated_branch(features) #512
        outputs_u = self.uncorrelated_branch(features) #512
        new_features = self.reconstruction_branch(torch.cat((outputs_m, outputs_u), dim=1))
        return outputs_m, outputs_u, new_features
    
    def calculate_margin(self, outputs_m):
        explain_loss = 0
        outputs_m = self.nn_sig(outputs_m)
        nor_outputs_m = F.normalize(outputs_m, p=2, dim=1)
        m_len = outputs_m.size(0)
        for i in range(m_len):
            x = nor_outputs_m[i,:]
            mean_value = torch.mean(x)
            reduced_vec = x[x > mean_value]
            #normalized_vec = reduced_vec / torch.norm(reduced_vec)
            margin_loss = reduced_vec - mean_value
            margin_loss = torch.sum(margin_loss).item() / (reduced_vec.size(0) + 1e-10)

            margin_loss = margin_loss if margin_loss > 0 else 0.
            explain_loss += margin_loss 
        explain_loss = explain_loss / (m_len + 1e-10) 
        return explain_loss

