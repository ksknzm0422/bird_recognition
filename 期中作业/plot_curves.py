# -*- encoding:utf-8 -*-
from __future__ import print_function, division

import torch
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets
from utils.transform import makeDefaultTransforms, makeNewTransforms
from utils.load import unpickle, list_subdirectories
from utils.net_cub import init_resnet50

def plot_loss_acc(model_name='resnet50', model_path=None):
    """绘制训练和测试损失及准确率"""
    root_dir = ''
    output_dir = model_path if model_path else os.path.join(root_dir, 'models', f'classification/{model_name}')
    model_history = os.path.join(output_dir, 'model_history.pkl')
    
    history = unpickle(model_history)
    history['train_acc'] = [tensor.cpu().item() for tensor in history['train_acc']]
    history['test_acc'] = [tensor.cpu().item() for tensor in history['test_acc']]
    
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], 'b-', label='Train Loss')
    plt.plot(history['test_loss'], 'r-', label='Test Loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training / Validation Loss - {model_name}')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], 'b-', label='Train Acc')
    plt.plot(history['test_acc'], 'r-', label='Test Acc')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training / Validation Accuracy - {model_name}')
    plt.legend()
    
    result_dir = f'result/figure/{model_name}'
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(f'{result_dir}/loss_acc.png')

def evaluate_model(model_name='resnet50', model_path=None, transform_method='new', stage='test'):
    """评估模型性能"""
    root_dir = ''
    data_root_dir = os.path.join(root_dir, 'CUB_200_2011', 'images_sorted')
    output_dir = model_path if model_path else os.path.join(root_dir, 'models', f'classification/{model_name}')
    model_file = os.path.join(output_dir, 'model_weights.pth')

    data_transforms = makeNewTransforms() if transform_method == 'new' else makeDefaultTransforms()
    image_dataset = datasets.ImageFolder(data_root_dir, data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=4)
    class_names = image_dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_resnet50(model_name)
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()

    labels_truth, labels_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels_truth.extend(labels.cpu().numpy())
            labels_pred.extend(preds.cpu().numpy())

    return np.array(labels_truth), np.array(labels_pred), class_names

def plot_metrics(labels_truth, labels_pred, class_names, model_name):
    """绘制分类报告及混淆矩阵"""
    class_report_df = pd.DataFrame(classification_report(labels_truth, labels_pred, target_names=class_names, output_dict=True))
    class_report_df.to_csv(f'result/csv/{model_name}_classification.csv', index=False)

    # 绘制精度、召回率和F1分数
    for metric in ['precision', 'recall', 'f1-score']:
        plt.figure(figsize=(10, 35))
        class_report_df.transpose()[metric][:-3].sort_values().plot(kind='barh')
        plt.xlabel(f'{metric.capitalize()} Score')
        plt.title(model_name)
        plt.grid(True)
        plt.savefig(f'result/figure/{model_name}/{metric.capitalize()}.png')

def plot_confusion_matrix(labels_truth, labels_pred, class_names, model_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels_truth, labels_pred)
    plt.figure(figsize=(40, 40))
    plt.imshow(cm, cmap='Reds')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.colorbar()
    plt.title(f'Confusion Matrix - Model {model_name}')
    plt.savefig(f'result/figure/{model_name}/Confusion_Matrix.png')

def main():
    with open('./models/final_model/config.json', 'r') as f:
        config = json.load(f)
    
    model_name = config['test']['model_name']
    model_path = config['test']['model_path']
    transform_method = config['test']['transform_method']
    
    labels_truth, labels_pred, class_names = evaluate_model(model_name, model_path, transform_method)
    plot_metrics(labels_truth, labels_pred, class_names, model_name)
    plot_confusion_matrix(labels_truth, labels_pred, class_names, model_name)
    plot_loss_acc(model_name, model_path)

if __name__ == "__main__":
    main()
