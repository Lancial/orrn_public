import os
import random
import torch
import torchio as tio
from torch.utils.data import Dataset

'''
modify this
'''
root = './'


'''
modify this
'''
SEQ_DATA_PATH = {
    '4d-lung-train': {
        'path': os.path.join(root, '4d-lung-train'),
        'format': 'meta'
    },
    'spare-train': {
        'path': os.path.join(root, 'spare-train'),
        'format': 'meta'
    },
}


def propogate_transform(subjects):
    '''
        parameters:
            data: list of torchio scalar image
            transform: torchio transform
        
        return: transformed series
    '''
    reproduced_transform = subjects[0].get_composed_history()
    for i in range(1, len(subjects)):
        subjects[i] = reproduced_transform(subjects[i])
    
    return subjects

class ZipDataset(Dataset):
    def __init__(self, names, dim, augmentation=None, sub_sequence=[i for i in range(10)], pair=False, size=(128, 128, 128)):
        
        self.dataset_lst = [ImageSequenceDataset(name, dim, augmentation, sub_sequence, pair, size=size) for name in names]

        self.lens = [len(dataset) for dataset in self.dataset_lst]
        
    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        for i, l in enumerate(self.lens):
            if idx < l:
                return self.dataset_lst[i][idx]
            idx = idx - l
        return None
    

class ImageSequenceDataset(Dataset):
    def __init__(self, name, dim, augmentation=None, sub_sequence=[i for i in range(10)], pair=False, size=(128, 128, 128)):
        self.name = name
        self.dataset = MetaDataset(SEQ_DATA_PATH[name], sub_sequence, pair)
        
        self.dim = dim
        preprocessing1 = tio.Compose([
            tio.Clamp(out_min=-1000, out_max=500),
            tio.RescaleIntensity((0, 1)),
            tio.Resize(target_shape=size)
        ])

        preprocessing2 = tio.Compose([
            tio.Clamp(out_min=0, out_max=0.02),
            tio.RescaleIntensity((0, 1)),
            tio.Resize(target_shape=size)
        ])
 
        if 'spare' in self.name:
            self.preprocess = preprocessing2
        else:
            self.preprocess = preprocessing1
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        tio_images = self.dataset[idx]
        subjects = [tio.Subject(image=img) for img in tio_images]

        subjects[0] = self.preprocess(subjects[0])
        if self.augmentation is not None:
            subjects[0] = self.augmentation(subjects[0])

        subjects = propogate_transform(subjects)

        tensor_images = [sub.image.tensor.float() for sub in subjects]
        data = torch.cat(tensor_images, dim=0)
        return data
    

class MetaDataset(Dataset):
    
    def __init__(self, data, sub_sequence, pair=False):
        self.dir_path = data['path']
        self.series = sorted([path for path, _, files in os.walk(self.dir_path) 
                  if any([('.mha' in c_file) and len(os.listdir(path)) > 1 and ('images' in path) for c_file in files])])
        self.sub_sequence = sub_sequence
        self.pair = pair
        
    def __len__(self):
        return len(self.series)
        
    def __getitem__(self, idx):
        serie = self.series[idx]
        
        f_list_in_serie = sorted([f for f in os.listdir(serie) if f.endswith('mha') and 'Mask' not in f])
        f_list_in_serie = [f_list_in_serie[i] for i in self.sub_sequence]
        vols = []
        if self.pair:
            samples = [0] + random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)
            tio_images = []
            for i in samples:
                f = f_list_in_serie[i]
                full_path = os.path.join(serie, f)
                tio_images.append(tio.ScalarImage(full_path))
            return tio_images
        
        for f in f_list_in_serie:
            full_path = os.path.join(serie, f)
            vols.append(tio.ScalarImage(full_path))

        tio_images = vols
        
        return tio_images