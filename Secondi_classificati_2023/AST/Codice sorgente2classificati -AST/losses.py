import torch
from torch import nn
from torch.nn import functional as F

class SupConLoss(nn.Module): # from : https://github.com/ilyassmoummad/scl_icbhi2017/blob/main/losses.py
    def __init__(self, temperature=0.06, device="cuda:0"): # temperature was not explored for this task
        #di fatto la temperatura è un parametro che controlla la scala della distribuzione nel softmax. PIù basso è più la distribuzione è sharp.
        super().__init__()
        self.temperature = temperature
        self.device = device




#calcola la perdita data una coppia di proiezioni.
    def forward(self, projection1, projection2, labels=None):

        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0] #troviamo il batch size

        #Le proiezioni vengono normalizzate e espanse e concatenate lungo una dimensione cosi da avere un risultato di un tensore di dim
        #[batch_size,2,feature_dim]

        if labels is None:
        
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
            #si crea una maschera con 1 nella diagonale e zero altrove, questo se non si hanno label dati in ingresso
        else:
            #la maschera che si crea se è fornita una label è binaria e indica se i campioni appartengono alla stessa classe.
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]  #numero di feature

        #torch.unbind rimuove una dimensione da un tensore e ritorna una tupla di slices nella dimensione specificata cioè 1.
        #in pratica se ho una matrice di shape(4,3) mi ritorna 3 tuple con 4 elementi  ognuno, cioè le varie colonne.
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) 



        #calcolo prodotto scalare fra le feature normalizzate per poi scalare in base al parametro temperatura.
        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)
        
        #si sottrae il massimo di ogni riga per migliorare la stabilità numerica del calcolo del softmax
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        #si crea una mask per escludere i logit delle stesse istanze, cioè self contrast.
        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0)
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask


        #calcolo probabilità logaritmica.
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        #calcolo media della probabilità logaritmica dei sample positivi.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #calcolo effettivo della loss.
        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss