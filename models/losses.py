import torch
import torch.nn.functional as F

def Ensemble_CE_Loss(logits, logits_mixed, features, features_mixed, y_a, y_b=None, 
                        gamma=0.5, lam=1.0, mixed_loss=True, weights=None, base_weight=2.0):
    #step 1: feature constrastive learning loss
    batch_size = logits.shape[0]
    features = features  / features.norm(dim=1, keepdim=True)
    features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
    #borrowed from CLIP
    features_logits = features @ features_mixed.t()
    #features_logits = features_logits.detach().clone()
    features_logits = features_logits.unsqueeze(2)

    #if weights is None:
    weights = features_logits
        #modulating_factor = 0
    #else:
    #    weights = torch.cat([weights, features_logits], dim=2)
    modulating_factor = torch.mean(weights, dim=2).view(batch_size, -1)
    modulating_factor = torch.softmax(modulating_factor, dim=-1)
    #features_pt = torch.softmax(features_logits, dim=-1)
    features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
    #step 2: supervised learning loss
    modulating_factor = modulating_factor.gather(1, features_ground_truth)

    if mixed_loss:
        logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits_mixed, dim=1)
        #logpt = modulating_factor**gamma * F.log_softmax(logits_mixed, dim=1)
        loss = lam*F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)
    else:
        logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits, dim=1)
        #logpt = modulating_factor**gamma * F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logpt, y_a)
    #print('CE_Loss=', loss.item(), "   feature_loss=", features_loss.item())
    return loss, weights

def Ensemble_CE_Loss2(logits, logits_mixed, features, features_mixed, y_a, y_b=None, 
                        gamma=0.5, lam=1.0, mixed_loss=True, weights=None, base_weight=2.0):
    #step 1: feature constrastive learning loss
    batch_size = logits.shape[0]
    #features = features / features.norm(dim=1, keepdim=True)
    #features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
    #borrowed from CLIP
    features_logits = features @ features_mixed.t()
    norm_factor = 1.0/torch.sqrt(1.0*features.shape[1])
    features_logits = features_logits*norm_factor
    #features_logits = features_logits.detach().clone()
    features_logits = features_logits.unsqueeze(2)

    if weights is None:
        weights = features_logits
        modulating_factor = 0
    else:
        modulating_factor = torch.mean(weights, dim=2).view(batch_size, -1)
        modulating_factor = torch.softmax(modulating_factor, dim=-1)
        #features_pt = torch.softmax(features_logits, dim=-1)
        features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
        #step 2: supervised learning loss
        modulating_factor = modulating_factor.gather(1, features_ground_truth)

    if mixed_loss:
        logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits_mixed, dim=1)
        #logpt = modulating_factor**gamma * F.log_softmax(logits_mixed, dim=1)
        loss = lam*F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)
    else:
        logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits, dim=1)
        #logpt = modulating_factor**gamma * F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logpt, y_a)
    #print('CE_Loss=', loss.item(), "   feature_loss=", features_loss.item())
    return loss, weights

def Ensemble_CE_Loss3(logits, logits_mixed, features, features_mixed, y_a, y_b=None, 
                        gamma=0.5, lam=1.0, mixed_loss=True, weights=None, base_weight=2.0):
    #step 1: feature constrastive learning loss
    batch_size = logits.shape[0]
    features = features  / features.norm(dim=1, keepdim=True)
    features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
    #borrowed from CLIP
    features_logits = features @ features_mixed.t()
    #features_logits = features_logits.detach().clone()
    features_logits = features_logits.unsqueeze(2)

    #if weights is None:
    weights = features_logits
        #modulating_factor = 0
    #else:
    #    weights = torch.cat([weights, features_logits], dim=2)
    modulating_factor = torch.mean(weights, dim=2).view(batch_size, -1)
    modulating_factor = torch.softmax(modulating_factor, dim=-1)
    #features_pt = torch.softmax(features_logits, dim=-1)
    features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
    #step 2: supervised learning loss
    modulating_factor = modulating_factor.gather(1, features_ground_truth)

    if mixed_loss:
        logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits_mixed, dim=1)
        #logpt = modulating_factor**gamma * F.log_softmax(logits_mixed, dim=1)
        loss = lam*F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)
    else:
        logpt = F.log_softmax(logits, dim=1)
        #logpt = modulating_factor**gamma * F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logpt, y_a)
    #print('CE_Loss=', loss.item(), "   feature_loss=", features_loss.item())
    return loss, weights