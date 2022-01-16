import torch
import torchvision

import pandas as pd
import numpy as np


def jaccard(y_true, y_pred):
    """ Jaccard a.k.a IoU score for batch of images
    """
    #Jaccarad = A intersection B / A union B
    #The closer index to 1, the better is the perfomance of the model
    num = y_true.size(0)
    #why do we introduce eps? to avoid division by 0 i guess ?
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    #I don't quite understand this parts
    intersection = (y_true_flat * y_pred_flat).sum(1)
    #THis neither
    union = ((y_true_flat + y_pred_flat) > 0.0).float().sum(1)
    
    score = (intersection) / (union + eps)
    score = score.sum() / num
    return score
    
#Dice coefficient = (2|X intersction Y|) / (|X| + |Y|)
#Similar to dice, but is considered semimetric(doesn't satisfy triangle inequality)
#gives less weight to outliers
def dice(y_true, y_pred):
    """ Dice a.k.a f1 score for batch of images
    """
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    intersection = (y_true_flat * y_pred_flat).sum(1)
    
    score =  (2 * intersection) / (y_true_flat.sum(1) + y_pred_flat.sum(1) + eps)
    score = score.sum() / num
    return score