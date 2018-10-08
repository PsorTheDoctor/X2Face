
# Load the datase

import sys
sys.path.append('./')
from frontalisation.frontalise_face import frontalise_face, frontalise_face_eyenosemouth


import torch.utils.data as data
import numpy as np
from torchvision.transforms import ToTensor, Scale, Compose, Pad, RandomHorizontalFlip, CenterCrop, RandomCrop, Resize
from PIL import Image
from ExtraTransforms import ColorJitter
from Utils import gkern, distance, dot

import torch
import scipy
import h5py
import os
import scipy.io
import scipy.signal
import cv2

from skimage import io, color

matfiletrain = '/scratch/local/ssd/ow/vc/cached_matlab_vc_train_all.mat'
matfileval = '/scratch/local/ssd/ow/vc/cached_matlab_vc_val_all.mat'
def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    return img

def load_img_mask(file_path):
	img = Image.open(file_path).convert('RGB')
	return img

def save_img(img, file_path):
    #img.save(file_path)
    pass


class VoxCeleb(data.Dataset):
	def __init__(self, num_views, random_seed, dataset, additional_face=False):
		super(VoxCeleb, self).__init__()
		self.additional_face = additional_face
		self.rng = np.random.RandomState(random_seed)
		if os.path.exists('/scratch/local/ssd/ow/faces/datasets/voxceleb/landmarks_samevideoimg_%d25thframe_5imgs_%d.npz' % (dataset, num_views)):
			files = np.load('/scratch/local/ssd/ow/faces/datasets/voxceleb/landmarks_samevideoimg_%d25thframe_5imgs_%d.npz' % (dataset, num_views))
			self.image_names = files['image_names']
			self.input_indices = files['input_indices']
			self.landmarks = files['landmarks']
			self.num_views = num_views
			self.transform = Compose([Scale((256,256)), ToTensor()])
			self.pose_transform = Compose([Scale((256,256)), ToTensor()])
			
			return
			
		
		# Load the matfile
		if dataset == 1:
			imdb = scipy.io.loadmat(matfiletrain)
		else:
			imdb = scipy.io.loadmat(matfileval)

		imdb = imdb['imdb']

		self.landmarks = imdb['images'][0][0][0]['label'][0][0]
		self.image_names = np.empty((imdb['image_names'][0][0].shape[1], 1), dtype='S100')
		indices = np.linspace(1, imdb['image_names'][0][0].shape[1], imdb['image_names'][0][0].shape[1])
		names = np.empty((imdb['image_names'][0][0].shape[1], 1), dtype='S100')
		video = np.empty((imdb['image_names'][0][0].shape[1], 1), dtype='S100')
		id = np.empty((imdb['image_names'][0][0].shape[1], 1), dtype=np.int32)
		for i in range(0, imdb['image_names'][0][0].shape[1]):
			if i % 1000 == 0:
				print('Split', i, dataset)

			temp_name = str(imdb['image_names'][0][0][0][i]).split('/')
			names[i] = temp_name[6]
			video[i] = temp_name[8]
			id[i] = int(temp_name[9][:-6])

			self.image_names[i] = str(imdb['image_names'][0][0][0][i][0]).replace('/users/koepke/data/voxceleb/faces/','/scratch/local/ssd/koepke/voxceleb/faces/')
			
			
			
		self.input_indices = np.zeros((imdb['image_names'][0][0].shape[1], num_views), dtype='int32')
		self.input_indices[:,0] = indices
		n_t = None
		v_t = None
		for i in range(0, imdb['image_names'][0][0].shape[1]):
			if i % 1000 == 0:
				print('Ind', i, dataset)
	
			n = names[i]
			v = video[i]
			t_id = id[i]

			if not((n == n_t) and (v == v_t)):

				same_names = names == n
				same_videos = video == v
				valid_indices = same_names & same_videos & (abs(id - t_id) < 5000)
				valid_indices[i] = 0

				valid_indices = np.squeeze(valid_indices);

			if indices[valid_indices > 0].shape[0] < num_views:
				ind_chosen = np.random.choice(indices[valid_indices > 0], num_views-1, replace=True)
			else:
				ind_chosen = np.random.choice(indices[valid_indices > 0], num_views-1, replace=False)
			n_t = n
			v_t = v
			self.input_indices[i,1:] = ind_chosen
		
		np.savez('/scratch/local/ssd/ow/faces/datasets/voxceleb/landmarks_samevideoimg_%d25thframe_5imgs_%d.npz' % (dataset, num_views), image_names=self.image_names, input_indices=self.input_indices, landmarks=self.landmarks)

		self.num_views = num_views
		self.transform = Compose([Scale((256,256)), ToTensor()])
		self.pose_transform = Compose([Scale((256,256)), ToTensor()])

	def __len__(self):
		return self.image_names.shape[0] - 1

	def __getitem__(self, index):
		if self.additional_face:
			(other_face, _) = self.get_blw_item(self.rng.randint(self.__len__()))

		else:
			other_face = []

		return self.get_blw_item(index)


	def get_eyebrow_heatmap_item(self, index):
		# Load the images
		i = self.num_views-1
		img_index = int(self.input_indices[index,i]) - 1
		# The facial features
		#pose_img, expr_img = generate_eyebrow_heatmap(landmarks, imgs[i+1].shape[0:2])
	        	
		#imgs[i+1] = pose_img.unsqueeze(0)
		#imgs[i+2] = expr_img.unsqueeze(0)
		poses, landmarks = self.get_blw_item(index)

		maskpath = self.image_names[img_index][0].replace('koepke/voxceleb/faces', 'ow/voxceleb/masks')
		maskpath = os.path.splitext(maskpath)[0] + '.png'
		dirpath = os.path.split(maskpath)[0]
		if not os.path.exists(maskpath):
		    if not os.path.exists(dirpath):
		        try:
		            os.makedirs(dirpath)
		        except OSError as err:
		            print(err)
		    mask_image = generate_eyesmouth_mask(landmarks)
		    save_img(mask_image, maskpath)
		    mask_image = self.pose_transform(mask_image)
		else:
		    mask_image = load_img_mask(maskpath)
		    mask_image = self.pose_transform(mask_image)
		    mask_image = mask_image[0:1,:,:]
		return (poses, landmarks), mask_image

	def get_blw_item(self, index):
		# Load the images
		imgs = [0] * (self.num_views+2)

		for i in range(0, self.num_views-1):
			img_index = int(self.input_indices[index,i]) - 1
			imgs[i] = load_img(str(self.image_names[img_index][0]))
			imgs[i] = self.transform(imgs[i])

		
		# Additional info
		i = self.num_views-1
		img_index = int(self.input_indices[index,i]) - 1
		imgs[i] = load_img(str(self.image_names[img_index][0]))
		imgs[i] = self.pose_transform(imgs[i])
		imgs[i+1] = np.array(load_img(str(self.image_names[img_index][0])))

		landmarks = self.landmarks[:,:,img_index].astype(np.int64).transpose()
		landmarks[:,0] = landmarks[:,0] / float(imgs[i+1].shape[1]) * 255.
		landmarks[:,1] = landmarks[:,1] / float(imgs[i+1].shape[0]) * 255.

		imgs[i+1] = imgs[i][0:1,:,:]
		imgs[i+2] = imgs[i][0:1,:,:]


		return imgs, landmarks

