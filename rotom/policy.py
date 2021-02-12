import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import get_lm


class FilterPolicyNet(nn.Module):
    """The model for filtering the augmented/unlabeled examples"""
    def __init__(self, num_classes=None, num_features=None, device='cuda'):
        super().__init__()
        self.device = device

        if num_features is None:
            self.num_features = num_classes * 2 # num_classes * 4
        else:
            self.num_features = num_features
        self.hidden_size = 10

        self.fc = nn.Linear(self.num_features, 1)
        self.fc_unlabeled = nn.Linear(self.num_features, 1)
        self.fc1 = nn.Linear(self.num_features, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def featurize(self, y, y_pred, y_aug):
        """Featurize an example given its predictions and labels"""
        KLD = torch.nn.KLDivLoss(reduction='none')
        return torch.cat((y, KLD(y_aug, y_pred)), dim=-1) # .mean(dim=-1).unsqueeze(-1)

    def forward(self, x, labeled=True):
        """Get probability of each examples"""
        if labeled:
            output = self.fc(x) # .sigmoid(output)
        else:
            output = self.fc_unlabeled(x)
        return output.squeeze()

class AugmentPolicyNetV4(nn.Module):

    def __init__(self, num_classes, device='cuda', lm='distilbert', bert_path=None):
        super().__init__()
        self.device = device
        self.bert = get_lm(lm=lm, bert_path=bert_path)
        if num_classes == 0:
            num_classes = 1
        self.num_classes = num_classes
        hidden_size = 768

        self.fc = nn.Linear(hidden_size, num_classes)
        self.hardness = nn.Linear(hidden_size, 1)

    def forward(self, x, y, x_enc=None, prediction=None):
        """Generate the random augment samples"""
        x = x.to(self.device)
        y = y.to(self.device)

        if x_enc is None:
            x_enc = self.bert(x)[0] # (bs, seq_len, hs)
            x_enc = x_enc[:, 0, :] # (bs, hs)

        if prediction is None:
            # return the uncertainty
            prob = self.fc(x_enc).softmax(dim=-1)
            mse = ((prob - y) ** 2).sum(dim=-1)
            return mse
        else:
            hard = self.hardness(x_enc).squeeze(-1).sigmoid()
            prob = prediction

            mse = ((prob - y) ** 2).mean(dim=-1)
            if len(y.size()) == 3: # tagging task
                mse = mse.mean(dim=-1) # ((prob - y) ** 2).mean(dim=[1,2])

            ind = mse + hard
            return ind / ind.sum() * x.size()[0]
