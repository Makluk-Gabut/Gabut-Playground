import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 1. PREPARASI DATA & AUGMENTASI (Standar Industri)
# Di dunia nyata, foto bisa miring, buram, atau terlalu terang. 
# Kita manipulasi gambar saat training supaya AI lebih tangguh di lapangan.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Balik kanan-kiri acak
        transforms.RandomRotation(15),     # Putar gambar acak up to 15 derajat
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Ubah pencahayaan acak
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalisasi standar ImageNet
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Hubungkan ke folder dataset kamu (Struktur folder: dataset/train/organik, dataset/train/anorganik)
data_dir = 'data/sampah_dataset' 
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes # Hasilnya: ['anorganik', 'organik']

# 2. LOAD PRE-TRAINED MODEL (Menggunakan Otak ResNet50)
# ResNet50 punya 50 layer yang sudah sangat pintar mengenali bentuk, tekstur, dan objek.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# "Freeze" semua layer awal agar bobot yang sudah pintar tidak rusak selama training
for param in model.parameters():
    param.requires_grad = False

# Ganti layer terakhir (Fully Connected Layer) khusus untuk problem kita (2 kelas: organik & anorganik)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) 
model = model.to(device)

# 3. OPTIMIZER & LOSS FUNCTION
criterion = nn.CrossEntropyLoss()
# Kita hanya mengoptimasi layer terakhir yang baru kita buat (model.fc)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 4. TRAINING LOOP DENGAN VALIDASI
num_epochs = 5
print(f"Memulai training di device: {device}\n" + "-"*30)

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Mode training
        else:
            model.eval()   # Mode evaluasi (validasi)

        running_loss = 0.0
        running_corrects = 0

        # Iterasi melewati seluruh gambar
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass + optimize hanya jika di fase training
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f"Epoch {epoch+1}/{num_epochs} | Bisnis {phase.upper()} -> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

print("\nTraining Selesai! Model siap digunakan untuk deploy.")

# 5. CARA SIMPAN MODEL UNTUK PRODUCTION
torch.save(model.state_dict(), 'model_sortir_sampah.pth')
print("Model berhasil disimpan ke 'model_sortir_sampah.pth'")
