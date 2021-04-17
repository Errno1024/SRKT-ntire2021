import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--start_from", type=int, default=0)

parser.add_argument("-c", "--checkpoint", type=str, default=None)
parser.add_argument("-t", "--train_teacher", action="store_true")
parser.add_argument("--train_dir", type=str, default="datasets/train")
parser.add_argument("--train_gt", type=str, default="datasets/groundtruth")
parser.add_argument("--teacher", type=str, default="model_teacher.pkl")
parser.add_argument("--checkpoint_name", type=str, default="model_kt_sr")

args = parser.parse_args()
distributed = args.local_rank >= 0
import os, sys

if not distributed:
    curdevice = 0
    gpu_start_from = args.start_from
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in range(args.start_from, args.start_from + args.gpus)])
else:
    curdevice = args.local_rank
    gpu_start_from = curdevice

from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm as _tqdm

class tqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        if curdevice == 0:
            self._tqdm = _tqdm(iterable, *args, **kwargs)
        else:
            self._tqdm = iterable
    def __iter__(self):
        return iter(self._tqdm)
    def __len__(self):
        return len(self._tqdm)
    def __getattr__(self, item):
        def default(*args, **kwargs):
            pass
        return getattr(self._tqdm, item, default)

CURDIR = os.path.dirname(os.path.abspath(sys.argv[0]))

from dataloader import dataset
import loss as losses

import torch.distributed

torch.cuda.set_device(curdevice)

def Log(message: str):
    if curdevice == 0:
        print(message)

from model import DehazeSR, Teacher

train_teacher = args.train_teacher
if train_teacher:
    model = Teacher()
else:
    model = DehazeSR()

model_teacher_name = args.teacher
model_checkpoint_name = args.checkpoint_name

checkpoint_file = args.checkpoint

if checkpoint_file is not None:
    _filepath = f"checkpoints/{checkpoint_file}"
    print(f'loading model from {_filepath}')
    state_dict = torch.load(_filepath, map_location='cpu')['model']

Log('setup')
if torch.cuda.is_available():
    if distributed:
        torch.distributed.init_process_group(backend='nccl')
    device = torch.device(curdevice)
else:
    if distributed:
        torch.distributed.init_process_group(backend='gloo')
    device = torch.device("cpu")

batch_size = 5
epochs = 400
train_mode = 100
iteration = 100
checkpoint_interval = 40
schedule = [i for i in range(4, epochs + 4, 4)]
learning_rate = 1.0e-4
lr_decay = 0.025
crop_size_w, crop_size_h = 16, 16

alpha = 1 # L1
beta = 0.3  # Laploss
gamma = 0.5 # Lab L2
delta = 1 # KT

batch_size *= args.gpus

model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.MultiStepLR(optimizer, schedule, 1 - lr_decay)

checkpoint_label = f"_{gpu_start_from}" if not distributed else ""

if distributed:
    model = torch.nn.parallel.DistributedDataParallel(model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )
else:
    class DistributedWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, input):
            return self.module(input)
    model = DistributedWrapper(model)

Log('loading dataset')
trainset = dataset(valid=False, verbose=True, train_dir=args.train_dir, gt_dir=args.train_gt, cropsize=(crop_size_w * 16, crop_size_h * 16))

if distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

dl_args = {}

if distributed:
    dl_args['sampler'] = train_sampler

dataloader = DataLoader(
    trainset,
    batch_size=batch_size,
    **dl_args,
)


if not train_teacher:
    Log('loading teacher')

    model_teacher = Teacher()
    model_teacher.load_state_dict(torch.load(f'checkpoints/{model_teacher_name}', map_location='cpu'))
    model_teacher.to(device)
    model_teacher.eval()

Log('training')

for epoch in range(epochs):
    if distributed:
        train_sampler.set_epoch(epoch)

    optimizer.zero_grad()
    model.zero_grad()
    gen = tqdm(desc='Epoch %03d' % epoch, postfix={'loss': float('inf')}, total=iteration)
    count = 0
    if epoch < train_mode:
        model.train()
    else:
        model.eval()
    while count < iteration:
        for train, gt in dataloader:
            if train_teacher:
                train = gt.to(device)
            else:
                train = train.to(device)
            gt = gt.to(device)

            if not train_teacher:
                with torch.no_grad():
                    _, tmid = model_teacher(gt)
                    del _
            pred, mid = model(train)
            while len(pred.shape) < 4:
                pred = pred.unsqueeze(0)
            pred_restrict = pred.clamp(0, 1)
            loss = 0

            l1 = losses.l1loss(pred_restrict, gt)
            lap = losses.laploss(pred_restrict, gt)
            lab = losses.lab2loss(pred_restrict, gt)
            if not train_teacher:
                lteacher = losses.l1loss(mid, tmid)

            loss = loss + alpha * l1
            loss = loss + beta * lap
            loss = loss + gamma * lab
            if not train_teacher:
                loss = loss + delta * lteacher

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            count += 1
            gen.update(1)

            postfix = {
                'LR': optimizer.state_dict()['param_groups'][0]['lr'],
                'L1': l1.item(),
                'Lap': lap.item(),
                'Lab': lab.item(),
            }
            if not train_teacher:
                postfix['KT'] = lteacher.item()
            gen.set_postfix(postfix)
            del train, gt, pred, loss, mid, l1, lap, lab, pred_restrict
            if not train_teacher:
                del tmid, lteacher
            if count >= iteration: break
    gen.close()
    optimizer.step()
    scheduler.step()
    if curdevice == 0:
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(
                {
                    'model': model.module.state_dict(),
                    'epoch': epoch,
                },
                'checkpoints/%s%s_%d.pkl' % (model_checkpoint_name, checkpoint_label, epoch),
            _use_new_zipfile_serialization=False)

if curdevice == 0:
    torch.save(
        {
            'model': model.module.state_dict(),
        },
        f'checkpoints/{model_checkpoint_name}{checkpoint_label}.pkl',
    _use_new_zipfile_serialization=False)

