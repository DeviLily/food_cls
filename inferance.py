import os
from os import path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import config as cfg
from model import resnet34
from utils import AverageMeter

class DatasetTest(Dataset):
    def __init__(self, root, data_path, transform=None):
        self.transform = transform
        
        self.data_path = path.join(root, data_path)
        self.filelist = os.listdir(self.data_path)
    
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        img_path = path.join(self.data_path, self.filelist[index])
        with open(img_path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform:
            sample = self.transform(sample)
        return sample

def main():
    if not torch.cuda.is_available():
        print('Plz train on cuda !')
        os._exit(0)
    
    # Construct Data Loader
    print('Load test dataset...')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    dataset = DatasetTest(cfg.root, 'data/food/test', transform=transform_test)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    print('%d test samples loaded...' % len(dataset))

    # Load best trained model
    print('Load best model...')
    best_filename = cfg.root + '/ckpt/model_best.pth.tar'
    state = torch.load(best_filename)
    print('Best model with acc@1: %.3f at epoch %d.' % (state['best_acc'], state['epoch']))
    model = resnet34()
    model.load_state_dict(state['state_dict_model'])

    if cfg.gpu is not None:
        print('Use cuda !')
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    
    # Inference
    print('Start testing...')
    pred = np.array([])

    model.eval()
    with torch.no_grad():
        for imgs in tqdm(loader):
            if cfg.gpu is not None:
                imgs = imgs.cuda(cfg.gpu, non_blocking=True)
            
            output = model(imgs)
            batch_pred = torch.argmax(output, dim=1)
            pred = np.append(pred, batch_pred.cpu().numpy())
    
    # Save output
    print('Saving...')
    with open(os.path.join(cfg.root, 'data/submission.txt'), 'w') as f:
        f.write('Id,Expected\n')
        for aline in zip(dataset.filelist, pred):
            f.write('%s,%d\n' % aline)
    
    print('Finish!')

if __name__ == '__main__':
    main()