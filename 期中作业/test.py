# -*- encoding:utf-8 -*-
from __future__ import print_function, division

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from utils.transform import makeDefaultTransforms, makeNewTransforms, makeAggresiveTransforms
from utils.net_cub import init_resnet50, init_net

def test(model_name=None, model_path=None, transform_method='new'):
    # 设置模型和数据目录
    root_dir = ''
    data_root_dir = os.path.join(root_dir, 'CUB_200_2011')
    data_dir = os.path.join(data_root_dir, 'images_sorted')

    # 设置模型路径
    if model_path is None:
        model_path = os.path.join(root_dir, 'models', 'classification', model_name)
    model_file = os.path.join(model_path, 'model_weights.pth')

    # 根据变换方法选择数据变换
    transform_map = {
        'default': makeDefaultTransforms,
        'new': makeNewTransforms
    }
    
    if transform_method not in transform_map:
        raise ValueError("只能选择 'new' 或 'default'")

    data_transforms = transform_map[transform_method]()
    image_datasets = {'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms)}
    dataloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=4, shuffle=True, num_workers=4)}
    class_names = image_datasets['test'].classes

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    if model_name is None:
        net_cub = init_resnet50(pretrained=True)
    else:
        net_cub = init_net(model_name=model_name)
        net_cub.load_state_dict(torch.load(model_file))
    
    net_cub.to(device)
    net_cub.eval()

    # 测试模型
    top5_correct = 0
    labels_truth, labels_pred, scores_pred = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net_cub(inputs)

            if isinstance(outputs, tuple):
                outputs, _ = outputs
            
            _, preds = torch.max(outputs, 1)
            _, top5_preds = torch.topk(outputs, 5, 1)

            top5_correct += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).any(dim=1).sum().item()

            labels_truth.extend(labels.cpu().numpy())
            labels_pred.extend(preds.cpu().numpy())
            scores_pred.append(outputs.cpu().numpy())

    # 计算准确率
    labels_truth = np.array(labels_truth)
    labels_pred = np.array(labels_pred)
    scores_pred = np.concatenate(scores_pred)
    
    test_accuracy = np.sum(labels_pred == labels_truth) / len(labels_truth)
    top5_accuracy = top5_correct / len(labels_truth)

    print('测试准确率:', test_accuracy)
    print('Top 5 测试准确率:', top5_accuracy)

# 读取配置文件并运行测试
with open('./models/final_model/config.json', 'r') as f:
    config = json.load(f)

model_path = config['test']['model_path']
model_name = config['test']['model_name']
transform_method = config['test']['transform_method']

test(model_name=model_name, model_path=model_path, transform_method=transform_method)
