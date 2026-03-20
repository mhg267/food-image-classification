import argparse
import cv2
import torch
import os
import torch.nn as nn
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt


from torchvision import transforms
from data_setup import FoodDataset
from model_architecture import Net
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from tqdm import tqdm




def get_args():
    parser = argparse.ArgumentParser('Model Inference')

    parser.add_argument("--best_model_path", "-m", type=str, default=None)
    parser.add_argument("--root", "-r", type=str, default=None)
    parser.add_argument("--image_path", "-p", type=str, default=None)
    parser.add_argument("--batch_size", "-b", type=int, default=16)

    args = parser.parse_args()

    return args

def test_model(data):
    model.eval()

    progress_bar = tqdm(data)
    predictions = []
    labels = []

    for (image, label) in progress_bar:
        image = image.to(device)
        label = label.to(device)

        labels.extend(label.cpu().tolist())

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1)
            predictions.extend(prediction.cpu().tolist())

    cm = confusion_matrix(labels, predictions, normalize='true')

    print("Accuracy: ", accuracy_score(labels, predictions))
    print("-------------------------------------------------")
    print("Classification Report:\n", classification_report(labels, predictions))
    print("-------------------------------------------------")
    print("Confusion Matrix:\n", cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)

    ax.set_title("Confusion Matrix - Food Classification")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(food_list)))
    ax.set_yticks(range(len(food_list)))
    ax.set_xticklabels(food_list, rotation=90, fontsize=6)
    ax.set_yticklabels(food_list, fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

def predict(image_path):
    original_image = cv2.imread(image_path)

    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = transforms.ToTensor()(image)
    image = image[None, :, :, :].to(device)

    model.eval()

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        softmax = nn.Softmax(dim=1)(output)
        probability = softmax[0, prediction].item() * 100

        print("Prediction of model is {} with {:.2f}%".format(food_list[prediction], probability))

        cv2.imshow("Prediction: {} Percentage: {:.2f}%".format(food_list[prediction], probability), original_image)
        cv2.waitKey(0)

if __name__ == '__main__':
    args = get_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])  # ImageNet mean and std
                                    ])

    test_dataset = FoodDataset(root=args.root,
                               train="test",
                               transform=transform
                               )

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 drop_last=False
                                 )

    food_list = os.listdir(args.root)
    food_list.sort()

    model = Net().to(device)

    if args.best_model_path is not None:
        checkpoint = torch.load(args.best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        print("No checkpoint file found! Please check and try again!")
        exit(0)

    test_model(test_dataloader)

    if args.image_path is not None:
        predict(args.image_path)
    else:
        print("No image path found! Please check and try again!")

