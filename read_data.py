## Modification from original HardNet implementation in
## https://raw.githubusercontent.com/DagnyT/hardnet/master/code/dataloaders/HPatchesDatasetCreator.py
## I need to clean it a little bit and modify some things, but it works

import os
import numpy as np
import cv2
import sys
import json
import keras
from tqdm import tqdm
import glob
import random

splits = ['a', 'b', 'c', 'view', 'illum']
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

class STNHPatches(keras.utils.Sequence):
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = ['ref','e1','e2','e3','e4','e5']
    def __init__(self, seqs, batch_size = 32):
        self.all_paths = []
        self.batch_size = batch_size
        self.dim = (32, 32)
        self.n_channels = 1
        self.sequences = {}
        self.sequences_n = {}
        for base in tqdm(seqs):
            name = base.split('/')
            self.name = name[-1]
            self.base = base
            for t in self.itr:
                im_path = os.path.join(base, t + '.png')
                img_n = cv2.imread(os.path.join(base, t + '.png'), 0)
                img   = cv2.imread(os.path.join(base, 'ref' + '.png'), 0)
                N = img.shape[0] / 32
                seq_im = np.array(np.split(img, N),
                                  dtype=np.uint8)
                seq_im_n = np.array(np.split(img_n, N),
                                    dtype=np.uint8)
                for i in range(int(N)):
                    path = os.path.join(base, t, str(i) + '.png')
                    self.all_paths.append(path)
                    self.sequences[path] = seq_im[i]
                    self.sequences_n[path] = seq_im_n[i]
        self.on_epoch_end()

    def get_images(self, index):
        path = self.all_paths[index]
        img = self.sequences[path].astype(np.float32)
        img_n = self.sequences_n[path].astype(np.float32)
        return img, img_n

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.all_paths) / self.batch_size))

    def __getitem__(self, index):
        img_clean = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim + (self.n_channels,))

        for i in range(self.batch_size):
            img, img_n = self.get_images(index*self.batch_size+i)
            img_clean[i] = np.expand_dims(img, -1)
            img_noise[i] = np.expand_dims(img_n, -1)

        return img_noise, img_clean

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        random.shuffle(self.all_paths)


class DenoiseHPatches(keras.utils.Sequence):
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self, seqs, batch_size = 32):
        self.all_paths = []
        self.batch_size = batch_size
        self.dim = (32, 32)
        self.n_channels = 1
        self.sequences = {}
        self.sequences_n = {}
        for base in tqdm(seqs):
            name = base.split('/')
            self.name = name[-1]
            self.base = base
            for t in self.itr:
                im_path = os.path.join(base, t + '.png')
                img_n = cv2.imread(im_path.replace('.png', '_noise.png'), 0)
                img = cv2.imread(im_path, 0)
                N = img.shape[0] / 32
                seq_im = np.array(np.split(img, N),
                                  dtype=np.uint8)
                seq_im_n = np.array(np.split(img_n, N),
                                    dtype=np.uint8)
                for i in range(int(N)):
                    path = os.path.join(base, t, str(i) + '.png')
                    self.all_paths.append(path)
                    self.sequences[path] = seq_im[i]
                    self.sequences_n[path] = seq_im_n[i]
        self.on_epoch_end()

    def get_images(self, index):
        path = self.all_paths[index]
        img = self.sequences[path].astype(np.float32)
        img_n = self.sequences_n[path].astype(np.float32)
        return img, img_n

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.all_paths) / self.batch_size))

    def __getitem__(self, index):
        img_clean = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim + (self.n_channels,))

        for i in range(self.batch_size):
            img, img_n = self.get_images(index*self.batch_size+i)
            img_clean[i] = np.expand_dims(img, -1)
            img_noise[i] = np.expand_dims(img_n, -1)

        return img_noise, img_clean

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        random.shuffle(self.all_paths)

class hpatches_sequence_folder:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self, base, noise=1):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        if noise:
            noise_path = '_noise'
        else:
            noise_path = ''
        for t in self.itr:
            im_path = os.path.join(base, t+noise_path+'.png')
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/32
            setattr(self, t, np.split(im, self.N))



def generate_triplets(labels, num_triplets, batch_size):
    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds
    triplets = []
    with open('tmp.txt','w') as f:
        import json
        json.dump(labels,f)
    indices = create_indices(np.asarray(labels))
    unique_labels = np.unique(np.asarray(labels))
    n_classes = unique_labels.shape[0]
    # add only unique indices in batch
    already_idxs = set()

    for x in tqdm(range(num_triplets)):
        if len(already_idxs) >= batch_size:
            already_idxs = set()
        c1 = np.random.randint(0, n_classes)
        while c1 in already_idxs:
            c1 = np.random.randint(0, n_classes)
        already_idxs.add(c1)
        c2 = np.random.randint(0, n_classes)
        while c1 == c2:
            c2 = np.random.randint(0, n_classes)
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1]))
            n2 = np.random.randint(0, len(indices[c1]))
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]))
        n3 = np.random.randint(0, len(indices[c2]))
        triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
    return np.array(triplets)

class HPatches():
    def __init__(self, train=True, transform=None, download=False, train_fnames=[],
                 test_fnames=[], denoise_model=None, use_clean=False):
        self.train = train
        self.transform = transform
        self.train_fnames = train_fnames
        self.test_fnames = test_fnames
        self.denoise_model = denoise_model
        self.use_clean = use_clean

    def set_denoise_model(self, denoise_model):
        self.denoise_model = denoise_model

    def denoise_patches(self, patches):
        batch_size = 100
        for i in tqdm(range(int(len(patches) / batch_size)), file=sys.stdout):
            batch = patches[i * batch_size:(i + 1) * batch_size]
            batch = np.expand_dims(batch, -1)
            batch = np.clip(self.denoise_model.predict(batch).astype(int),
                                        0, 255).astype(np.uint8)[:,:,:,0]
            patches[i*batch_size:(i+1)*batch_size] = batch
        batch = patches[i*batch_size:]
        batch = np.expand_dims(batch, -1)
        batch = np.clip(self.denoise_model.predict(batch).astype(int),
                        0, 255).astype(np.uint8)[:,:,:,0]
        patches[i*batch_size:] = batch
        return patches

    def read_image_file(self, data_dir, train = 1):
        """Return a Tensor containing the patches
        """
        if self.denoise_model and not self.use_clean:
            print('Using denoised patches')
        elif not self.denoise_model and not self.use_clean:
            print('Using noisy patches')
        elif self.use_clean:
            print('Using clean patches')
        sys.stdout.flush()
        patches = []
        labels = []
        counter = 0
        hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
        if train:
            list_dirs = self.train_fnames
        else:
            list_dirs = self.test_fnames

        for directory in tqdm(hpatches_sequences, file=sys.stdout):
           if (directory in list_dirs):
            for tp in tps:
                if self.use_clean:
                    sequence_path = os.path.join(data_dir, directory, tp)+'.png'
                else:
                    sequence_path = os.path.join(data_dir, directory, tp)+'_noise.png'
                image = cv2.imread(sequence_path, 0)
                h, w = image.shape
                n_patches = int(h / w)
                for i in range(n_patches):
                    patch = image[i * (w): (i + 1) * (w), 0:w]
                    patch = cv2.resize(patch, (32, 32))
                    patch = np.array(patch, dtype=np.uint8)
                    patches.append(patch)
                    labels.append(i + counter)
            counter += n_patches

        patches = np.array(patches, dtype=np.uint8)
        if self.denoise_model and not self.use_clean:
            print('Denoising patches...')
            patches = self.denoise_patches(patches)
        return patches, labels


class DataGeneratorDesc(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, data, labels, num_triplets = 1000000, batch_size=50, dim=(32,32), n_channels=1, shuffle=True):
        # 'Initialization'
        self.transform = None
        self.out_triplets = True
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data = data
        self.labels = labels
        self.num_triplets = num_triplets
        self.on_epoch_end()

    def get_image(self, t):
        def transform_img(img):
            if self.transform is not None:
                img = transform(img.numpy())
            return img

        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a).astype(float)
        img_p = transform_img(p).astype(float)
        img_n = transform_img(n).astype(float)

        img_a = np.expand_dims(img_a, -1)
        img_p = np.expand_dims(img_p, -1)
        img_n = np.expand_dims(img_n, -1)
        if self.out_triplets:
            return img_a, img_p, img_n
        else:
            return img_a, img_p

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.triplets) / self.batch_size))

    def __getitem__(self, index):
        y = np.zeros((self.batch_size, 1))
        img_a = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_p = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        if self.out_triplets:
            img_n = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        for i in range(self.batch_size):
            t = self.triplets[self.batch_size*index + i]
            img_a_t, img_p_t, img_n_t = self.get_image(t)
            img_a[i] = img_a_t
            img_p[i] = img_p_t
            if self.out_triplets:
                img_n[i] = img_n_t

        return {'a': img_a, 'p': img_p, 'n': img_n}, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.triplets = generate_triplets(self.labels, self.num_triplets, 32)

#####################################################

def generate_triplets_regularised(labels, num_triplets, batch_size):

    triplets = []

    for x in tqdm(range(num_triplets)):

        im1 = random.choice([key for key in labels])
        im2 = random.choice([key for key in labels])

        tp1 = random.choice([ key for key in labels[im1] ])
        tp2 = random.choice([ key for key in labels[im1] ])
        tp3 = random.choice([ key for key in labels[im2] ])

        idx1 = random.choice(labels[im1][tp1])
        idx2 = random.choice(labels[im2][tp3])

        penalty = 1

        if im1 == im2:
            penalty = 0.75

        if im1 == im2 and idx1 == idx2:
            penalty = 0

        triplets.append([ (im1,tp1,idx1), (im1,tp2,idx1), (im2,tp3,idx2), penalty ])

    return np.array(triplets)

class HPatchesRegularised():
    def __init__(self, train=True, transform=None, download=False, train_fnames=[],
                 test_fnames=[], denoise_model=None, use_clean=False):
        self.train = train
        self.transform = transform
        self.train_fnames = train_fnames
        self.test_fnames = test_fnames
        self.denoise_model = denoise_model
        self.use_clean = use_clean

    def read_image_file(self, data_dir, train = 1):
        """Return a Tensor containing the patches
        """
        if self.denoise_model and not self.use_clean:
            print('Using denoised patches')
        elif not self.denoise_model and not self.use_clean:
            print('Using noisy patches')
        elif self.use_clean:
            print('Using clean patches')
        sys.stdout.flush()
        patches = {}
        labels = {}
        counter = 0
        hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
        if train:
            list_dirs = self.train_fnames
        else:
            list_dirs = self.test_fnames

        for directory in tqdm(hpatches_sequences, file=sys.stdout):
            if (directory in list_dirs):
                labels[directory] = {}
                patches[directory] = {}
                for tp in tps:
                    labels[directory][tp] = []
                    patches[directory][tp] = []
                    if self.use_clean:
                        sequence_path = os.path.join(data_dir, directory, tp)+'.png'
                    else:
                        sequence_path = os.path.join(data_dir, directory, tp)+'_noise.png'
                    image = cv2.imread(sequence_path, 0)
                    h, w = image.shape
                    n_patches = int(h / w)
                    for i in range(n_patches):
                        patch = image[i * (w): (i + 1) * (w), 0:w]
                        patch = cv2.resize(patch, (32, 32))
                        patch = np.array(patch, dtype=np.uint8)
                        patches[directory][tp].append(patch)
                        labels[directory][tp].append(i)

        return patches, labels

class DataGeneratorDescRegularised(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, data, labels, num_triplets = 1000000, batch_size=50, dim=(32,32), n_channels=1, shuffle=True):
        # 'Initialization'
        self.transform = None
        self.out_triplets = True
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data = data
        self.labels = labels
        self.num_triplets = num_triplets
        self.on_epoch_end()
        tmp = self.__getitem__(0)
        print(tmp[0]['alpha'].shape)

    def get_image(self, t):
        def transform_img(img):
            if self.transform is not None:
                img = transform(img.numpy())
            return img

        im1 = t[0][0]
        im2 = t[2][0]

        tp1 = t[0][1]
        tp2 = t[1][1]
        tp3 = t[2][1]

        idx1 = t[0][2]
        idx2 = t[2][2]

        a, p, n = self.data[im1][tp1][idx1], self.data[im1][tp2][idx1], self.data[im2][tp3][idx2]

        img_a = transform_img(a).astype(float)
        img_p = transform_img(p).astype(float)
        img_n = transform_img(n).astype(float)

        img_a = np.expand_dims(img_a, -1)
        img_p = np.expand_dims(img_p, -1)
        img_n = np.expand_dims(img_n, -1)
        if self.out_triplets:
            return img_a, img_p, img_n
        else:
            return img_a, img_p

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.triplets) / self.batch_size))

    def __getitem__(self, index):
        y = np.zeros((self.batch_size, 1))
        img_a = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_p = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        alpha = np.zeros(self.batch_size)
        if self.out_triplets:
            img_n = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        for i in range(self.batch_size):
            t = self.triplets[self.batch_size*index + i]
            img_a_t, img_p_t, img_n_t = self.get_image(t)
            img_a[i] = img_a_t
            img_p[i] = img_p_t
            if self.out_triplets:
                img_n[i] = img_n_t
            alpha[i] = t[3]

        alpha = np.expand_dims(alpha, -1)
        return {'a': img_a, 'p': img_p, 'n': img_n, 'alpha': alpha}, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.triplets = generate_triplets_regularised(self.labels, self.num_triplets, 32)

if __name__ == '__main__':
    hpatches_dir = './hpatches'
    splits_path = './splits.json'

    splits_json = json.load(open(splits_path, 'rb'))
    split = splits_json['a']

    train_fnames = split['train']
    test_fnames = split['test']

    seqs = glob.glob(hpatches_dir+'/*')
    seqs = [os.path.abspath(p) for p in seqs]
    seqs_train = list(filter(lambda x: x.split('/')[-1] in train_fnames, seqs))
    seqs_test = list(filter(lambda x: x.split('/')[-1] in split['test'], seqs))

    ### Descriptor loading and training
    # Loading images
    hPatches = HPatchesRegularised(train_fnames=train_fnames, test_fnames=test_fnames,
                        use_clean=False)
    # Creating training generator
    training_generator = DataGeneratorDescRegularised(*hPatches.read_image_file(hpatches_dir, train=1), num_triplets=100000, batch_size=500)
    # Creating validation generator
    val_generator = DataGeneratorDescRegularised(*hPatches.read_image_file(hpatches_dir, train=0), num_triplets=10000, batch_size=500)
