import os

from sklearn.model_selection import train_test_split

def data_split(root):
    cls_list = os.listdir(root)
    cls_list.sort()

    dataset = []
    for label, cls in enumerate(cls_list):
        cls_path = os.path.join(root, cls)
        for image_path in os.listdir(cls_path):
            image = os.path.join(cls_path, image_path)
            dataset.append((image, label))

    labels_train = [lab for _, lab in dataset]

    train, test = train_test_split(dataset,
                                   test_size=0.2,
                                   random_state=42,
                                   stratify=labels_train
                                   )

    labels_test = [lab for _, lab in test]
    test, val = train_test_split(test,
                                 test_size=0.5,
                                 random_state=42,
                                 stratify=labels_test
                                 )

    return train, val, test







