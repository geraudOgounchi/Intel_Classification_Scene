from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = '/kaggle/input/datasets/puneet6060/intel-image-classification/seg_train/seg_train'
VAL_DIR  = '/kaggle/input/datasets/puneet6060/intel-image-classification/seg_test/seg_test'
IMG_SIZE = 150
BATCH    = 32


def get_data():
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(VAL_DIR,  transform=val_transforms)

    # num_workers=4 et pin_memory=True pour accélérer le chargement GPU
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,
                                  num_workers=4, pin_memory=True,
                                  persistent_workers=True)
    test_dataloader  = DataLoader(val_dataset,   batch_size=BATCH, shuffle=False,
                                  num_workers=4, pin_memory=True,
                                  persistent_workers=True)
    return train_dataloader, test_dataloader
