from UnwrappedFace import UnwrappedFaceWeightedAverage

import shutil
import os
import numpy as np
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Scale, Compose
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from VoxCelebData_withmask import VoxCeleb, FramesDataset

parser = argparse.ArgumentParser(description='UnwrappedFace')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--sampler_lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--threads', type=int, default=1, help='Num Threads')
parser.add_argument('--batchSize', type=int, default=16, help='Batch Size')
parser.add_argument("--dataset", default='data/nemo', help="Path to dataset")
parser.add_argument("--folder", default="out", help="out folder")
parser.add_argument("--arch", default='unet_64', help="Network architecture")
parser.add_argument("--num_iter", default=10000, type=int, help="Number of iterations")

args = parser.parse_args()


def run_batch(imgs, requires_grad=False, volatile=False, other_images=None):
    for i in range(0, len(imgs)):
        imgs[i] = Variable(imgs[i], requires_grad=requires_grad, volatile=volatile).cuda()

    poses = imgs[-3]
    print('Poses', poses.size())

    if not other_images is None:
        for i in range(0, len(other_images)):
            other_images[i] = Variable(other_images[i], requires_grad=requires_grad, volatile=volatile).cuda()

        return (model(poses, *imgs[0:-3]), model(poses, *other_images[0:-3])), imgs + [poses], other_images[0:-3]

    return model(poses, *imgs[0:-3]), imgs + [poses]


model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=512)
model = model.cuda()

parameters = [{'params': model.parameters()}]
optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

train_set = FramesDataset(args.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=16, shuffle=True)

if not os.path.exists(args.folder):
    os.makedirs(args.folder)

for j in trange(args.num_iter):
    batch = next(iter(training_data_loader))
    imgs = Variable(batch['video_array'], requires_grad=True).cuda()
    result = model(imgs[:, 0], imgs[:, 1])

    loss = torch.abs(result - imgs[:, 0]).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if j % 100 == 0:
        torchvision.utils.save_image(imgs[:, 1].data.cpu(), '%s/inp_%d.png' % (args.folder, j))        
        torchvision.utils.save_image(result.data.cpu(), '%s/result_%d.png' % (args.folder, j))
        torchvision.utils.save_image(imgs[:, 0].data.cpu(), '%s/gt_%d.png' % (args.folder, j))

torch.save(model.state_dict(), os.path.join(args.folder, "model.cpk"))

