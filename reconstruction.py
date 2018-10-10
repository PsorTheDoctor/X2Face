import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import imageio
import argparse
from VoxCelebData_withmask import FramesDataset
from UnwrappedFace import UnwrappedFaceWeightedAverage


def reconstruction_loss(a, b):
    return torch.abs(a - b).mean()


def reconstruction(generator, checkpoint, log_dir, dataset, format='.gif'):
    log_dir = os.path.join(log_dir, 'reconstruction')

    checkpoint = torch.load(checkpoint)
    generator.load_state_dict(checkpoint)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    loss_list = []
    generator.eval()
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            imgs = x['video_array']
            apperance = x['video_array'][:, 0]

            results = []
            for i in range(imgs.size()[1]):
                result = generator(apperance, i)
                results.append(result.unsqueeze(dim=1))

            results = torch.cat(results, dim=1)
            loss_list.append(reconstruction_loss(imgs, results))

            results = results.data.cpu().numpy()
            results = results[0].transpose((0, 2, 3, 1))
            imageio.mimsave(os.path.join(log_dir, x['name'] + format), results)

    print ("Reconstruction loss: %s" % np.mean(loss_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UnwrappedFace')
    parser.add_argument("--dataset", default='data/nemo', help="Path to dataset")
    parser.add_argument("--folder", default="out", help="out folder")
    parser.add_argument("--arch", default='unet_64', help="Network architecture")
    parser.add_argument("--format", default='.gif', help="Save format")
    args = parser.parse_args()


    model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=512)
    model = model.cuda()

    dataset = FramesDataset(args.dataset, is_train=False)
    reconstruction(model, os.path.join(args.folder, 'model.cpk'), args.folder, dataset, args.format)
