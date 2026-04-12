import torch.nn as nn
import torch.nn.functional as F


class IntelCNN_PyTorch(nn.Module):
    def __init__(self, num_classes=6):
        super(IntelCNN_PyTorch, self).__init__()
        # Block 1
        self.conv1a = nn.Conv2d(3,   32,  kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(32,  32,  kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(32)
        self.drop1  = nn.Dropout2d(0.1)
        # Block 2
        self.conv2a = nn.Conv2d(32,  64,  kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(64,  64,  kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(64)
        self.drop2  = nn.Dropout2d(0.2)
        # Block 3
        self.conv3a = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3    = nn.BatchNorm2d(128)
        self.drop3  = nn.Dropout2d(0.3)
        # Block 4
        self.conv4  = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4    = nn.BatchNorm2d(256)
        self.drop4  = nn.Dropout2d(0.3)
        # Classifier
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc1     = nn.Linear(256, 256)
        self.fc2     = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1 — BN appliqué après les deux convolutions
        x = F.relu(self.conv1a(x))
        x = F.relu(self.bn1(self.conv1b(x)))
        x = self.drop1(F.max_pool2d(x, 2))
        # Block 2
        x = F.relu(self.conv2a(x))
        x = F.relu(self.bn2(self.conv2b(x)))
        x = self.drop2(F.max_pool2d(x, 2))
        # Block 3
        x = F.relu(self.conv3a(x))
        x = F.relu(self.bn3(self.conv3b(x)))
        x = self.drop3(F.max_pool2d(x, 2))
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop4(F.max_pool2d(x, 2))
        # Classifier
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)   # logits bruts


def get_tensorflow_model(img_size=150, num_classes=6):
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(img_size, img_size, 3))

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.1)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.6)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inp, out)
