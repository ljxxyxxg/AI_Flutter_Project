import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments

# 데이터 경로 설정
train_dir = 'path/to/train'
test_dir = 'path/to/test'

# 데이터 전처리
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# 데이터셋 로드
train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 설정
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(train_dataset.classes)
)

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 손실 함수 및 옵티마이저 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 학습 함수 정의
def train(epoch, model, train_loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).logits
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    
    print(f'Epoch {epoch}: Loss = {total_loss/len(train_loader)}, Accuracy = {correct/len(train_loader.dataset)}')

# 평가 함수 정의
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).logits
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    
    print(f'Test Accuracy = {correct/len(test_loader.dataset)}')

# 학습 및 평가 루프
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, optimizer)
    evaluate(model, test_loader)

# 모델 저장
model.save_pretrained('./vit_model')
