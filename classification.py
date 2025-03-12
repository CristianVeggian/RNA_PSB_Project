import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from models.LeNet5 import LeNet5
from models.VGG16 import VGG16
from models.VGG19 import VGG19
import numpy as np
import random

import pickle as pkl

#para medir o tempo de treino e teste
import time

#especificar seeds para reprodutibilidade
def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_classification(model_name='LeNet5', database='csp_db_1', num_epochs=10, batch_size=64, seed=42):

    set_seed(seed)

    if model_name not in ['LeNet5', 'VGG16', 'VGG19']:
        print('Modelo Selecionado não existe')
        exit()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = os.path.join("databases", database)

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"Classes detectadas: {dataset.classes}")

    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    print(f"Tamanho do lote: {images.shape}") 
    print(f"Rótulos do lote: {labels}")

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size 

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total de imagens: {len(dataset)}")
    print(f"Treino: {len(train_dataset)} | Validação: {len(val_dataset)} | Teste: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)  
    match model_name:
        case 'LeNet5':
            model = LeNet5(num_classes).to(device)
        case 'VGG16':
            model = VGG16(num_classes).to(device)
        case 'VGG19':
            model = VGG19(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loss_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_loss_list.append(val_loss)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        print(f"Época {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}")

    end_time = time.time()

    elapsed_train_time = end_time - start_time

    torch.save(model.state_dict(), os.path.join('models', 'pre_trained', f"{model_name}_{database}.pth"))

    start_time = time.time()

    model.eval()
    test_correct = 0
    test_total = 0

    labels_pkl = []
    predicted_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            labels_pkl.extend(labels.cpu().numpy())
            predicted_list.extend(predicted.cpu().numpy())

            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    end_time = time.time()

    elapsed_test_time = end_time - start_time

    test_accuracy = test_correct / test_total

    pkl.dump({'y_true':labels_pkl}, file = open(os.path.join('labels', f'labels_{database}'), "wb"))

    print(f'\nTempo de treino da rede neural: {elapsed_train_time:.6f}')
    print(f'Tempo de teste da rede neural: {elapsed_test_time:.6f}')
    print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")

    return {"tempo_treino":elapsed_train_time,
            "tempo_teste":elapsed_test_time,
            "acuracia_test":test_accuracy,
            "loss":loss,
            "val_loss":val_loss_list,
            "predicoes":predicted_list}