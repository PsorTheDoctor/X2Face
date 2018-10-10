import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import imageio
import argparse
from VoxCelebData_withmask import FramesDataset, PairedDataset
from UnwrappedFace import UnwrappedFaceWeightedAverage
from torch.autograd import Variable

def transfer(generator, checkpoint, log_dir, dataset, format='.gif', number_of_pairs=100):
    log_dir = os.path.join(log_dir, 'transfer')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint = torch.load(checkpoint)
    generator.load_state_dict(checkpoint)

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=number_of_pairs)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=1)

    generator.eval()
    for it, x in tqdm(enumerate(dataloader)):
        imgs = Variable(x['first_video_array'], volatile=True).cuda()
        apperance = Variable(x['second_video_array'][:, 0, :, :, :], volatile=True).cuda()

        results = []
        for i in range(imgs.size()[1]):
            result = generator(imgs[:, i], apperance)
            results.append(result.unsqueeze(dim=1))

        results = torch.cat(results, dim=1)
        results = results.data.cpu().numpy()
        results = results[0].transpose((0, 2, 3, 1))
        results = (results * 255).astype('uint8')        
        img_name = "-".join([x['first_name'][0], x['second_name'][0]]) + format
        imageio.mimsave(os.path.join(log_dir, img_name), results)


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
    transfer(model, os.path.join(args.folder, 'model.cpk'), args.folder, dataset, args.format)
