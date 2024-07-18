from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper
from dassl.data.samplers import build_sampler
from dassl.utils import Registry, check_availability
from dassl.data.transforms import INTERPOLATION_MODES, build_transform
from dassl.data.datasets import DATASET_REGISTRY,build_dataset
from dassl.utils import read_image
from dassl.data.data_manager import build_data_loader

from collections import defaultdict
import torch


def build_dataset_XU(cfg, pred=[], prob=[]):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg, pred, prob)


def build_data_loader_XU(
        cfg,
        sampler_type="SequentialSampler",
        data=None,
        data_source=None,
        batch_size=64,
        n_domain=0,
        n_ins=2,
        tfm=None,
        is_train=True,
        dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper
        # Build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
        )
        assert len(data_loader) > 0

    else:
        dataset_wrapper = DatasetWrapper_XU
        # Build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
        )

    return data_loader



class Dataloader_eval(DataManager):
    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        super().__init__(cfg, custom_tfm_train, custom_tfm_test, dataset_wrapper)
        dataset = build_dataset(cfg)
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )
        self.train_loader_x = train_loader_x





class Dataloader_XU(DataManager):
    def __init__(self,
                 cfg,
                 custom_tfm_train=None,
                 custom_tfm_test=None,
                 dataset_wrapper=None,
                 pred=[],
                 prob=[]):

        # Load dataset
        dataset = build_dataset_XU(cfg, pred, prob)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x/clean data
        train_loader_x = build_data_loader_XU(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data=dataset,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=DatasetWrapper_XU
        )

        # Build train_loader_u/noisy data
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader_XU(
                cfg,
                sampler_type=sampler_type_,
                data=dataset,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader_XU(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data=dataset,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader_XU(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data=dataset,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)


class DatasetWrapper_XU(DatasetWrapper):
    def __init__(self, cfg, data, data_source, transform=None, is_train=False):
        super().__init__(cfg, data_source, transform, is_train)
        self.probability_x = data.probability_x

    def __getitem__(self, idx):
        item = self.data_source[idx]
        prob = self.probability_x[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "prob_x": prob,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output



def split_dataset_by_label(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into class-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)

    for item in data_source:
        output[item.label].append(item)

    return output
