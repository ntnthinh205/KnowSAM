import os
import logging
import h5py
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import patients_to_slices

class build_Dataset(Dataset):
    def __init__(self, args, data_dir, split, transform=None, labeled_slice=None, model="None"):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.model = model
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        self.args = args

        if self.split == "train":
            labeled_path = os.path.join(self.data_dir + "/labeled/image")
            sample_list_labeled = os.listdir(labeled_path)
            sample_list_labeled = [os.path.join(labeled_path, item) for item in sample_list_labeled]
            self.sample_list = sample_list_labeled
            print("train total {} samples".format(len(self.sample_list)))
        elif self.split == "train_semi":
            labeled_path = os.path.join(self.data_dir + "/labeled/image")
            unlabeled_path = os.path.join(self.data_dir + "/unlabeled/image")
            sample_list_labeled = os.listdir(labeled_path)
            sample_list_unlabeled = os.listdir(unlabeled_path)
            self.sample_list_labeled = [os.path.join(labeled_path, item) for item in sample_list_labeled]
            self.sample_list_unlabeled = [os.path.join(unlabeled_path, item) for item in sample_list_unlabeled]
            self.sample_list = self.sample_list_labeled + self.sample_list_unlabeled
            print("train total {} labeled samples, {} unlabeled samples".
                  format(len(sample_list_labeled), len(sample_list_unlabeled)))
        elif self.split == "val":
            val_path = os.path.join(self.data_dir + "/val/image")
            sample_list_val = os.listdir(val_path)
            self.sample_list = [os.path.join(val_path, item) for item in sample_list_val]
            print("val total {} samples".format(len(self.sample_list)))
        elif self.split == "train_semi_list":
            labeled_path = os.path.join(self.data_dir + "/train.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + "/images/" + image_name for image_name in self.image_list]
            self.sample_list_labeled = patients_to_slices(args.dataset, args.labeled_num)

            print("train total {} samples".format(len(self.sample_list)))
        elif self.split == "val_semi_list":
            labeled_path = os.path.join(self.data_dir + "/val.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + "/images/" + image_name for image_name in self.image_list]
            print("val total {} samples".format(len(self.sample_list)))
        elif self.split == "train_acdc_list":
            labeled_path = os.path.join(self.data_dir + "/train_slices.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/slices/" + image_name + ".h5" for image_name in self.image_list]
            # filter out missing files to avoid DataLoader worker crashes
            missing = [p for p in self.sample_list if not os.path.exists(p)]
            if missing:
                logging.warning("train_acdc_list: %d missing HDF5 files will be skipped. Example: %s", len(missing), missing[:3])
                self.sample_list = [p for p in self.sample_list if os.path.exists(p)]
            self.sample_list_labeled = patients_to_slices(args.dataset, args.labeled_num)
            print("train total {} samples".format(len(self.sample_list)))
        elif self.split == "val_acdc_list":
            labeled_path = os.path.join(self.data_dir + "/val.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/" + image_name + ".h5" for image_name in self.image_list]
            # filter missing validation files and warn (prevents worker FileNotFoundError)
            missing = [p for p in self.sample_list if not os.path.exists(p)]
            if missing:
                logging.warning("val_acdc_list: %d missing HDF5 files in %s/data; they will be skipped. Example: %s", len(missing), self.data_dir, missing[:3])
                self.sample_list = [p for p in self.sample_list if os.path.exists(p)]
            print("val total {} samples".format(len(self.sample_list)))
        elif self.split == "test_acdc_list":
            labeled_path = os.path.join(self.data_dir + "/test.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/" + image_name + ".h5" for image_name in self.image_list]
            missing = [p for p in self.sample_list if not os.path.exists(p)]
            if missing:
                logging.warning("test_acdc_list: %d missing HDF5 files will be skipped. Example: %s", len(missing), missing[:3])
                self.sample_list = [p for p in self.sample_list if os.path.exists(p)]
            print("test total {} samples".format(len(self.sample_list)))
        elif "test" in self.split:
            if "CVC-300" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/CVC-300/image")
            elif "CVC-ClinicDB" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/CVC-ClinicDB/image")
            elif "CVC-ColonDB" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/CVC-ColonDB/image")
            elif "ETIS-LaribPolypDB" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/ETIS-LaribPolypDB/image")
            elif "Kvasir" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/Kvasir/image")
            elif "ISIC2018" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/image")
            elif "DDTI" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/DDTI/image")
            elif "tn3k" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/tn3k/image")
            elif "BrainMRI" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/image")
            elif "MRI_Hippocampus" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/image")
            else:
                test_path = None

            if test_path:
                print('test_path: ', test_path)
                sample_list_val = os.listdir(test_path)
                self.sample_list = [os.path.join(test_path, item) for item in sample_list_val]
                print("test total {} samples".format(len(self.sample_list)))
            else:
                if "BCSS" in self.split:
                    test_list_path = os.path.join(self.data_dir + "/test.list")

                with open(test_list_path, 'r') as f:
                    self.image_list = f.readlines()
                self.image_list = [item.replace('\n', '') for item in self.image_list]
                self.sample_list = [self.data_dir + "/images/" + image_name for image_name in self.image_list]
                print("test total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        if "_list" not in self.split:
            case = self.sample_list[idx]
            image = cv2.cvtColor(cv2.imread(case), cv2.COLOR_BGR2RGB)
            ori_image = cv2.cvtColor(cv2.resize(image.copy(), (256, 256)), cv2.COLOR_RGB2BGR)
            # image = (image - self.pixel_mean) / self.pixel_std
            image = image / 255.0
            image = image.astype(np.float32)
            label_path = case.replace("image", "mask")

            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) / 255
            if "val" in self.split or "test" in self.split:
                if self.transform:
                    data = self.transform(image=image, mask=label)
                    image = data['image']
                    label = data['mask']
            else:
                if self.transform:
                    if idx < len(self.sample_list_labeled):
                        data = self.transform["train_weak"](image=image, mask=label)
                        image = data['image']
                        label = data['mask']
                    else:
                        data = self.transform["train_strong"](image=image, mask=label)
                        image = data['image']
                        label = data['mask']

            label[label < 0.5] = 0
            label[label > 0.5] = 1
            image = image.transpose(2, 0, 1).astype('float32')
            ori_image = ori_image.transpose(2, 0, 1).astype('float32')
            image, label = torch.tensor(image), torch.tensor(label)
            if "test" in self.split:
                sample = {"image": image, "label": label, "ori_image": ori_image}
            else:
                sample = {"image": image, "label": label,}
            return sample

        else:
            if "val" not in self.split and "test" not in self.split:
                case = self.sample_list[idx]
                h5f = h5py.File(case)
                image = h5f['image'][:].astype(np.float32)
                label = h5f['label'][:].astype(np.float32)
                if self.transform:
                    if idx < self.sample_list_labeled:
                        data = self.transform["train_weak"](image=image, mask=label)
                        image = data['image']
                        label = data['mask']
                    else:
                        data = self.transform["train_strong"](image=image, mask=label)
                        image = data['image']
                        label = data['mask']

                image = np.asarray(image)
                if image.ndim == 2:
                    image = image[np.newaxis, ...]
                elif image.ndim == 3:
                    if image.shape[-1] in (1, 3):
                        image = np.transpose(image, (2, 0, 1))
                    elif image.shape[0] not in (1, 3):
                        raise ValueError(f"Unexpected image shape {image.shape} for {case}")
                else:
                    raise ValueError(f"Unexpected image ndim {image.ndim} for {case}")

                image = torch.tensor(image, dtype=torch.float32)
                label = torch.tensor(label, dtype=torch.float32)
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                sample = {"image": image, "label": label, "idx": idx}
                return sample
            else:
                case = self.sample_list[idx]
                h5f = h5py.File(case)
                image = h5f['image'][:].astype('float32')
                label = h5f['label'][:].astype('float32')

                image, label = torch.tensor(image), torch.tensor(label)
                if self.model != "CAML":
                    sample = {"image": image, "label": label}
                else:
                    sample = {"image": image, "label": label, "idx": idx}
                return sample





