# -*- encoding:utf-8 -*-
from __future__ import print_function, division
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
from multiprocessing.dummy import freeze_support
from utils.train import train_model
from utils.transform import makeNewTransforms
from utils.data_sort import data_sort
from utils.set_seed import set_seed
from net_train import alexnet_cub_train, resnet18_cub_train

def resnet50_pre_train(start='pretrain', num_epochs=80, batch_size=32, lr_new=1e-2, lr_fine=1e-3,
                       transform_method='new', device=None, dir="./models/", model_name=None):
    root_dir = 'CUB_200_2011'
    data_dir = os.path.join(root_dir, 'images_sorted')
    
    if model_name is None:
        model_name = f'resnet50_lrnew={lr_new}_lrfine={lr_fine}_transform:{transform_method}' if start == 'pretrain' \
            else f'resnet50_from_scratch_lr={lr_new}_transform:{transform_method}'
    
    working_dir = os.path.join(dir, model_name)
    os.makedirs(working_dir, exist_ok=True)
    
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    
    if transform_method == 'new':
        data_transforms = makeNewTransforms()
    else:
        raise ValueError("transform_method must be 'new'.")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    
    from utils.net_cub import init_resnet50  # Import here to keep dependencies minimal
    resnet_cub = init_resnet50(pretrained=(start == 'pretrain'))
    
    optimizer = optim.SGD(resnet_cub.parameters(), lr=lr_new, momentum=0.9, weight_decay=1e-4) if start == 'scratch' \
        else SGD_different_lr(resnet_cub, lr_finetune=lr_fine, lr_new=lr_new, weight_decay=1e-4)

    resnet_cub.to(device)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    model_ft = train_model(model=resnet_cub, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                           device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes, num_epochs=num_epochs,
                           working_dir=working_dir, log_history=True)

    torch.save(model_ft.state_dict(), os.path.join(working_dir, 'model_weights.pth'))

    config = {
        "train": {
            "lr_new": lr_new,
            "lr_fine": lr_fine,
            "batch_size": batch_size,
            "transform_method": transform_method,
            "num_epoch": num_epochs,
            "dir": dir,
            "model_name": model_name,
            "train_set": start
        },
        "test": {
            "model_path": working_dir,
            "model_name": model_name,
            "transform_method": transform_method
        }
    }

    with open(os.path.join(working_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def main():
    if not os.path.exists('./CUB_200_2011/images_sorted'):
        data_sort()
    set_seed(27)   
    with open('./models/final_model/config2.json', 'r') as f:
        config = json.load(f)['train']
    
    lr_new = config['lr_new']
    lr_fine = config['lr_fine']
    batch_size = config['batch_size']
    num_epochs = config['num_epoch']
    working_dir = config['dir']
    model_name = config['model_name']
    transform_method = config['transform_method']
    train_set = config['train_set']
    
    print(f"lr_fine={lr_fine}, lr_new={lr_new}, epoch={num_epochs}")

    resnet50_pre_train(train_set, num_epochs=num_epochs, lr_new=lr_new, lr_fine=lr_fine,
                       batch_size=batch_size, transform_method=transform_method,
                       model_name=model_name, dir=working_dir)

if __name__ == '__main__':
    freeze_support()
    main()
