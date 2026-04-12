import torch
import argparse
from utils import prep
from models.cnn import IntelCNN_PyTorch
from models.train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle CNN")
    parser.add_argument('--framework', type=str, choices=['pytorch', 'tensorflow'],
                        default='pytorch', help="Framework à utiliser (default: pytorch)")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Nombre d'époques d'entraînement")
    parser.add_argument('--lr',  type=float, default=0.001,  help="Learning rate")
    parser.add_argument('--wd',  type=float, default=0.0001, help="Weight decay")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="Mode : 'train' ou 'eval' (default: train)")
    parser.add_argument('--cuda', action='store_true', help="Utiliser le GPU si disponible")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.framework == 'tensorflow':
        run_tensorflow(args)
    else:
        run_pytorch(args)


def run_pytorch(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] Device: {device}")

    # Récupération des données
    train_dataloader, test_dataloader = prep.get_data()

    # Création du validation set (split)
    from torch.utils.data import random_split, DataLoader

    dataset = train_dataloader.dataset
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_dataloader.batch_size,
                                  shuffle=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=train_dataloader.batch_size,
                                shuffle=False)

    # Modèle
    model = IntelCNN_PyTorch().to(device)

    if args.mode == 'eval':
        model.load_state_dict(torch.load("geraud_model.pth", map_location=device))
        print("Model loaded from geraud_model.pth")

    # Trainer corrigé
    trainer = Trainer(model,
                      train_dataloader,
                      val_dataloader,
                      test_dataloader,
                      args.lr,
                      args.wd,
                      args.epochs,
                      device)

    if args.mode == 'train':
        trainer.train(save=True, plot=True)

    # Évaluation finale CORRIGÉE
    trainer.test()


def run_tensorflow(args):
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks
    from models.cnn import get_tensorflow_model

    print(f"[TensorFlow] GPUs: {tf.config.list_physical_devices('GPU')}")

    IMG_SIZE = 150
    BATCH    = 32
    DATA_DIR = '/kaggle/input/datasets/puneet6060/intel-image-classification/seg_train/seg_train'
    VAL_DIR  = '/kaggle/input/datasets/puneet6060/intel-image-classification/seg_test/seg_test'
    AUTOTUNE = tf.data.AUTOTUNE

    norm    = layers.Rescaling(1./255)
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH)
    val_ds   = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, shuffle=False)

    train_ds = train_ds.map(
        lambda x, y: (norm(tf.clip_by_value(augment(x, training=True), 0, 255)), y),
        num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (norm(x), y),
        num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    model = get_tensorflow_model(IMG_SIZE)

    if args.mode == 'eval':
        model = tf.keras.models.load_model("geraud_model.keras")
        print("Model loaded from geraud_model.keras")
    else:
        model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        cb = [
            callbacks.ModelCheckpoint('geraud_model.keras', save_best_only=True,
                                       monitor='val_accuracy', verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                    restore_best_weights=True, verbose=1),
        ]

        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb)
        print("Model saved to geraud_model.keras")

    loss, acc = model.evaluate(val_ds, verbose=1)
    print(f"\nTest Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")


if __name__ == '__main__':
    main()
