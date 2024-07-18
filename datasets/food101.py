import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
import random
import numpy as np

@DATASET_REGISTRY.register()
class Food101(DatasetBase):
    dataset_dir = "food-101"

    def __init__(self, cfg, pred=[], prob=[]):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.meta_dir = os.path.join(self.dataset_dir, "meta")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            classnames = []
            with open(os.path.join(self.meta_dir, "classes.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    classnames.append(line.strip())
            cname2lab = {c: i for i, c in enumerate(classnames)}

            trainval = self.read_data(cname2lab, "train.txt")
            train, val = self.split_data(trainval, 0.8)
            test = self.read_data(cname2lab, "test.txt")
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        if len(pred) > 0:
            pred_idx = pred.nonzero()[0]
            train_x = [train[i] for i in pred_idx]
            self.probability_x = [prob[i] for i in pred_idx]
            print("clean data has a size of %d" % (len(train_x)))

            pred_idx = (1 - pred).nonzero()[0]
            train_u = [train[i] for i in pred_idx]
            self.probability_u = [prob[i] for i in pred_idx]
            print("noisy data has a size of %d" % (len(train_u)))

        else:
            train_x = train
            train_u = None

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)
        label_set = set()
        for item in train:
            label_set.add(item.label)
        self._num_classes = max(label_set) + 1


    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.meta_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname = line + ".jpg"
                line = line.split("/")
                classname = line[0]
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def split_data(self, trainval, p_train):
        print(f"Splitting trainval into {p_train:.0%} train and {1-p_train:.0%} val")
        random.seed(1)
        np.random.seed(1)
        random.shuffle(trainval)
        n_total = len(trainval)
        n_train = round(n_total * p_train)
        return trainval[:n_train], trainval[n_train:]