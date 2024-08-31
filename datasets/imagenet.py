import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .add_noise import generate_fewshot_dataset_with_symflip_noise, generate_fewshot_dataset_with_pairflip_noise

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg, pred=[], prob=[]):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                if cfg.DATASET.FP_TYPE == "symflip":
                    train = generate_fewshot_dataset_with_symflip_noise(train, num_shots=num_shots,
                                                                        num_fp=cfg.DATASET.NUM_FP, seed=seed)
                elif cfg.DATASET.FP_TYPE == "pairflip":
                    train = generate_fewshot_dataset_with_pairflip_noise(train, num_shots=num_shots,
                                                                         num_fp=cfg.DATASET.NUM_FP, seed=seed)
                else:
                    raise ValueError(f"There is no such type of noise!")
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

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

        super().__init__(train_x=train_x, train_u=train_u, val=test, test=test)
        label_set = set()
        for item in train:
            label_set.add(item.label)
        self._num_classes = max(label_set) + 1

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
