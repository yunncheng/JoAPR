import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, read_json, write_json

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

@DATASET_REGISTRY.register()
class Food101N(DatasetBase):
    dataset_dir = "food-101N"
    dataset_dir_food101 = "food-101"

    def __init__(self, cfg, pred=[], prob=[]):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.dataset_dir_food101 = os.path.join(root, self.dataset_dir_food101)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.image_dir_food101 = os.path.join(self.dataset_dir_food101, "images")
        self.meta_dir = os.path.join(self.dataset_dir_food101, "meta")
        self.split_path = os.path.join(self.dataset_dir, "split_guo_Food101N.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            def _convert(items, path_prefix):
                out = []
                for impath, label, classname in items:
                    impath = os.path.join(path_prefix, impath)
                    item = Datum(impath=impath, label=int(label), classname=classname)
                    out.append(item)
                return out

            print(f"Reading split from {self.split_path}")
            split = read_json(self.split_path)
            train = _convert(split["train"], self.image_dir)
            val = _convert(split["val"], self.image_dir)
            test = _convert(split["test"], self.image_dir_food101)

        else:
            classnames = []
            with open(os.path.join(self.meta_dir, "classes.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    classnames.append(line.strip())
            cname2lab = {c: i for i, c in enumerate(classnames)}
            train, val, _ = DTD.read_and_split_data(self.image_dir, 0.2, 0.05, 0.001)

            test = self.read_data(cname2lab, "test.txt")

            self.save_split(train, val, test, self.split_path, self.image_dir, self.image_dir_food101)

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
                impath = os.path.join(self.image_dir_food101, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items


    def save_split(self, train, val, test, filepath, path_prefix, path_prefix_food101):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                impath = impath.replace(path_prefix_food101, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")
