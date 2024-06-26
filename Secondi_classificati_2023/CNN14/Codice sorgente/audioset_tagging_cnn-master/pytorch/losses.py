import torch
import torch.nn.functional as F
from torch import nn

def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    

class SupConLoss(nn.Module): # from : https://github.com/ilyassmoummad/scl_icbhi2017/blob/main/losses.py
    def __init__(self, temperature=0.06, device="cuda:0"): # temperature was not explored for this task
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, projection1, projection2, labels=None):
        

        print(projection1)
        print(projection2)
        if not isinstance(projection1, torch.Tensor):
            projection1 = torch.tensor(projection1).to(self.device)
        if not isinstance(projection2, torch.Tensor):
            projection2 = torch.tensor(projection2).to(self.device)

        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0)
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss