# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.utils.data
import random
from data.base_data_loader import BaseDataLoader
from data import online_dataset_for_photos as RGB_EVENT


def CreateDataset(opt):
    dataset = RGB_EVENT.EventHDR_Dataset()
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        print(self.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            dataset = self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
