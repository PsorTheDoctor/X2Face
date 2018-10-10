import torch.utils.data as data
import numpy as np
from torchvision.transforms import ToTensor, Scale, Compose
from PIL import Image


def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    return img


class VoxCeleb(data.Dataset):
    def __init__(self, num_views, random_seed, dataset):
        super(VoxCeleb, self).__init__()
        files = np.load('/scratch/local/ssd/ow/faces/datasets/voxceleb/landmarks_samevideoimg_%d25thframe_5imgs_%d.npz' % (dataset, num_views))
        self.image_names = files['image_names']
        self.input_indices = files['input_indices']
        self.landmarks = files['landmarks']
        self.num_views = num_views
        self.transform = Compose([Scale((256,256)), ToTensor()])

    def __len__(self):
        return self.image_names.shape[0] - 1

    def __getitem__(self, index):
        return self.get_blw_item(index)

    def get_blw_item(self, index):
        # Load the images
        imgs = [0] * self.num_views

        for i in range(0, self.num_views-1):
            img_index = int(self.input_indices[index,i]) - 1
            imgs[i] = load_img(str(self.image_names[img_index][0]))
            imgs[i] = self.transform(imgs[i])

        return imgs


import os
import warnings
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
from itertools import permutations


class VideoToTensor(object):
    """Convert video array to Tensor."""
    def __call__(self, video_array):
        video_array = np.array(video_array, dtype='float32')
        return {'video_array': video_array.transpose((0, 3, 1, 2))}


class SelectRandomFrames(object):
    """Select frames from video, to make a batch.
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, reflect_pad_time=True, consequent=False):
        self.reflect_pad_time = reflect_pad_time
        self.consequent = consequent
        self.number_of_frames = 2

    def set_number_of_frames(self, number_of_frames):
        self.number_of_frames = number_of_frames

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images for selection
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: List of number_of_frames images
        """
        if self.reflect_pad_time:
            clip = np.concatenate([clip, clip[::-1]], axis=0)
        frame_count = clip.shape[0]

        num_frames_to_select = self.number_of_frames
        if self.consequent:
            first_frame = np.random.choice(max(1, frame_count - num_frames_to_select + 1), size=1)[0]
            selected = clip[first_frame:(first_frame + self.number_of_frames)]
        else:
            selected_index = np.sort(np.random.choice(range(frame_count), replace=False, size=num_frames_to_select))
            selected = clip[selected_index]

        return selected


class AllAugmentationTransform:
    def __init__(self):
        self.transforms = []
        self.transforms.append(SelectRandomFrames())
        self.transforms.append(VideoToTensor())

    def set_number_of_frames(self, number_of_frames):
        self.select.set_number_of_frames(number_of_frames)

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class FramesDataset(Dataset):
    """Dataset of videos, represented as image of consequent frames"""
    def __init__(self, root_dir, augmentation_param, image_shape=(64, 64, 3), is_train=True,
                 random_seed=0, classes_list=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.image_shape = tuple(image_shape)
        self.classes_list = classes_list

        assert os.path.exists(os.path.join(root_dir, 'test'))
        print("Use predefined train-test split.")
        train_images = os.listdir(os.path.join(root_dir, 'train'))
        test_images = os.listdir(os.path.join(root_dir, 'test'))
        self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')


        if is_train:
            self.images = train_images
        else:
            self.images = test_images

        if is_train:
            self.transform = AllAugmentationTransform(**augmentation_param)
        else:
            self.transform = VideoToTensor()

    def __len__(self):
        return len(self.images)

    def set_number_of_frames_per_sample(self, number_of_frames):
        self.transform.set_number_of_frames(number_of_frames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        if img_name.endswith('.png') or img_name.endswith('.jpg'):
            image = io.imread(img_name)

            if len(image.shape) == 2 or image.shape[2] == 1:
                image = gray2rgb(image)

            if image.shape[2] == 4:
                image = image[..., :3]

            image = img_as_float32(image)

            video_array = np.moveaxis(image, 1, 0)

            video_array = video_array.reshape((-1, ) + self.image_shape)
            video_array = np.moveaxis(video_array, 1, 2)
        elif img_name.endswith('.gif') or img_name.endswith('.mp4'):
            video = np.array(mimread(img_name))
            if video.shape[-1] == 4:
                video = video[..., :3]
            video_array = img_as_float32(video)
        else:
            warnings.warn("Unknown file extensions  %s" % img_name, Warning)

        out = self.transform(video_array)
        #add names
        out['name'] = os.path.basename(img_name)

        return out


class PairedDataset(Dataset):
    """
    Dataset of pairs.
    """
    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        classes_list = self.initial_dataset.classes_list

        if classes_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx,:ny].reshape(2, -1).T
        else:
            images = self.initial_dataset.images
            name_to_index = {name:index for index, name in enumerate(images)}
            classes = pd.read_csv(classes_list)
            classes = classes[classes['name'].isin(images)]
            names = classes['name']
            labels = classes['cls']

            name_pairs = []
            for cls in np.unique(labels):
                name_pairs += list(permutations(names[labels == cls], 2))

            xy = []
            for first, second in name_pairs:
                xy.append((name_to_index[first], name_to_index[second]))

            xy = np.array(xy)

        number_of_pairs = min(xy.shape[0], number_of_pairs)
        np.random.seed(seed)
        self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]

        first = {'first_' + key: value for key, value in first.items()}
        second = {'second_' + key: value for key, value in second.items()}

        return {**first, **second}
