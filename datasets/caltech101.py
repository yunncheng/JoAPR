import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
from .add_noise import generate_fewshot_dataset_with_symflip_noise,generate_fewshot_dataset_with_pairflip_noise


IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg, pred=[], prob=[]):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
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
                if cfg.DATASET.FP_TYPE == "symflip":
                    train = generate_fewshot_dataset_with_symflip_noise(train, num_shots=num_shots,
                                                                         num_fp=cfg.DATASET.NUM_FP, seed=seed)
                elif cfg.DATASET.FP_TYPE == "pairflip":
                    train = generate_fewshot_dataset_with_pairflip_noise(train, num_shots=num_shots,
                                                                         num_fp=cfg.DATASET.NUM_FP, seed=seed)
                else:
                    raise ValueError(f"There is no such type of noise!")
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

