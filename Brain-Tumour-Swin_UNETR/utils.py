import os
import json
import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum/self.count, self.sum)

def datafold_read(datalist, basedir, fold=0, key='training'):
    with open(datalist) as f:
        json_data = json.load(f)
    
    json_data = json_data[key]
    
    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if 'fold' in d and d['fold'] == fold:
            val.append(d)
        else:
            tr.append(d)
    return tr, val

def save_checkpoint(model, epoch, filename='model.pt', best_acc=0, dir_add='C:/Users/mekha/Desktop/Brain-Tumour-Detection/Brain-Tumour-Swin_UNETR/results'):
    state_dict = model.state_dict()
    save_dict = {'epoch': epoch, 'best_acc': best_acc, 'state_dict': state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print('saving checkpoint', filename)


