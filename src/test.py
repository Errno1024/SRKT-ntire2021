import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional
import os
import argparse

from dataloader import testset
from forward import x8
from model import DehazeSR, Wrapper

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='The folder where hazy images are stored.')
parser.add_argument('-o', '--output', type=str, default='.', help='The output path.')

args = parser.parse_args()

def main(args):
    if not os.path.isdir(args.dataset):
        raise ValueError(f'{args.dataset} is not a valid folder.')

    if not os.path.isdir(args.output):
        raise ValueError(f'{args.output} is not a valid folder.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DehazeSR()

    checkpoint = "SRKT.pkl"
    tset = testset(args.dataset, verbose=True)

    model.load_state_dict(torch.load(checkpoint)['model'])

    model = Wrapper(model)
    model.eval()
    model.to(device)

    loader = DataLoader(
        tset,
        batch_size=1,
    )

    with torch.no_grad():
        gen = tqdm(loader)
        for valid, filename in gen:
            filename = filename[0]
            valid = valid.to(device)
            try:
                pred = x8(valid, model=model).clamp(0.0, 1.0)
            except Exception as e:
                print(f"{filename}:", e)
                continue
            img = torchvision.transforms.functional.to_pil_image(pred.squeeze(0))
            img.save(f'{args.output}/{filename}')

            del valid, pred

if __name__ == '__main__':
    main(args)
