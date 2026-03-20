import argparse
import torch
import os

from data_setup import FoodDataset
from model_architecture import Net

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report



class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_args():
    parser = argparse.ArgumentParser(description='model arguments')

    parser.add_argument('--root', '-r', type=str, default='/Users/minhhung/Documents/Code/Python/Computer Vision/Data/Dataset/Food Classification Dataset', help='data root')
    parser.add_argument('--epochs', '-e', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='number of workers')
    parser.add_argument('--lr', '-l', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--tensorboard', '-t', type=str, default='tensorboard', help='tensorboard logging')
    parser.add_argument("--trained_model", "-tm", type=str, default="trained_model")
    parser.add_argument("--checkpoint", "-cp", type=str, default='trained_model/last_model.pth')

    args = parser.parse_args()

    return args




if __name__ == '__main__':

    args = get_args()

    print("Epoch number: ", args.epochs)
    print("Batch size: ", args.batch_size)
    print("-------------------------------")
    print("Learning rate: ", args.lr)


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225])  # ImageNet mean and std
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])  # ImageNet mean and std
    ])

    early_stopping = EarlyStopping(patience=20, min_delta=1e-3)


    if not os.path.exists(args.trained_model):
        os.makedirs(args.trained_model)

    tensorboard_writer = SummaryWriter(log_dir=args.tensorboard)

    train_dataset = FoodDataset(root=args.root,
                                train="train",
                                transform=train_transform
                                )

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=True
                              )

    validation_dataset = FoodDataset(root=args.root,
                               train="val",
                               transform=val_transform
                               )

    validation_loader = DataLoader(validation_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             drop_last=False
                                   )

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    train_iter_size = len(train_loader)
    val_iter_size = len(validation_loader)

    best_loss = float('inf')
    best_acc = 0

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        epoch_start = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        epoch_start = 0
        best_loss = float('inf')
        best_acc = 0

    for epoch in range(epoch_start, args.epochs):
        model.train()
        progress_bar_train = tqdm(train_loader)

        train_outputs = []
        train_labels = []
        train_loss_sum = 0.0
        total_train_samples = 0

        for iter, (images, labels) in enumerate(progress_bar_train):
            images, labels = images.to(device), labels.to(device)
            train_labels.extend(labels.cpu().tolist())

            # Forward
            train_output = model(images)
            train_pred = torch.argmax(train_output, dim=1)
            train_outputs.extend(train_pred.cpu().tolist())

            train_loss = criterion(train_output, labels)
            train_loss_value = train_loss.item()
            train_loss_sum += train_loss_value * images.size(0)
            total_train_samples += images.size(0)
            train_acc = accuracy_score(train_labels, train_outputs)

            progress_bar_train.set_description("Epoch {}/{} || Iteration {}/{} || Loss: {:.4f} || Accuracy_train: {:.4f}".format(epoch+1, args.epochs,
                                                                                                     iter+1, train_iter_size,
                                                                                                     train_loss_value, train_acc)
                         )

            tensorboard_writer.add_scalar("Loss/train_iteration", train_loss_value, (epoch*train_iter_size+iter))

            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        avg_train_loss = train_loss_sum / total_train_samples

        tensorboard_writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch+1)
        tensorboard_writer.add_scalar("Accuracy/train_epoch", train_acc, epoch+1)


        model.eval()

        progress_bar_val = tqdm(validation_loader)
        val_outputs = []
        val_labels = []
        val_loss_sum = 0.0
        total_val_samples = 0

        for iter, (images, labels) in enumerate(progress_bar_val):
            images, labels = images.to(device), labels.to(device)
            val_labels.extend(labels.cpu().tolist())

            with torch.no_grad():
                val_output = model(images)
                val_pred = torch.argmax(val_output, dim=1)
                val_outputs.extend(val_pred.cpu().tolist())

                val_loss = criterion(val_output, labels)
                val_loss_value = val_loss.item()
                val_loss_sum += val_loss_value * images.size(0)
                total_val_samples += images.size(0)
                val_acc = accuracy_score(val_labels, val_outputs)

                tensorboard_writer.add_scalar("Loss/val_iteration", val_loss_value, (epoch*val_iter_size+iter))

                progress_bar_val.set_description("Loss: {:.4} || Accuracy_val: {:.4}".format(val_loss_value,
                                                                                     val_acc)
                                         )

        avg_val_loss = val_loss_sum / total_val_samples
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Current LR: {current_lr:.6f}")

        tensorboard_writer.add_scalar("LR", current_lr, epoch+1)
        tensorboard_writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch+1)
        tensorboard_writer.add_scalar("Accuracy/val_epoch", val_acc, epoch+1)

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"Train | Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val | Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}"
        )

        checkpoint = {
            'epoch': epoch+1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'best_acc': best_acc
        }

        torch.save(checkpoint, f"{args.trained_model}/last_model.pth")

        if (best_loss > avg_val_loss) or (best_acc < val_acc):
            best_loss = avg_val_loss
            best_acc = val_acc

            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc
            }

            torch.save(checkpoint, f"{args.trained_model}/best_model.pth")

        if (epoch+1) % 5 == 0:
            print(classification_report(val_labels, val_outputs))


        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break




