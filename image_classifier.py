import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.utils.data
import os

# Define data transformations for data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomCrop(224), transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # RGB =  3 channnel, Standard deviation
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# Define the data directory
data_dir = 'dataset'
# Create data loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_size)

class_names = image_datasets['train'].classes
print(class_names)

# Load the pre-trained ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for name, param in model.named_parameters():
    if "fc" in name: # Unfreeze the final classification layer
        param.requires_grad = True
    else:
        param.requires_grad = False
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

# Training loop
nums_epochs = 50
for epoch in range(nums_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrcet = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrcet += torch.sum(preds == labels.data)
        epoch_loss = running_loss / dataset_size[phase]
        epoch_acc = running_corrcet.double() / dataset_size[phase]

        print(f'{phase} Loss: {epoch_loss: .4f} Acc: {epoch_acc: .4f}')
print("Training END")

# Save the model
torch.save(model.state_dict(), 'flower_classification_model.pth')
