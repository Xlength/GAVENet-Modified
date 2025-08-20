import os
import multiprocessing
import sys
import json
import random
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
from Config import cfg as config
from Hub.trainer import R2Vessels
from Hub.dataclass import VesselsDataset
from Tool.transformation import (
    random_affine, random_hsv, random_vertical_flip, random_horizontal_flip,
    to_torch_tensors, pad_images_unet, random_cutout
)
from Tool.data_util import SubsetRandomSampler, SubsetSequentialSampler, get_folds



torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True



def create_dataloaders(
        dataset, data, train_idx, test_idx, transforms_train,
        transforms_test
):
    dataset_train = dataset(data, transforms.Compose(transforms_train))
    dataset_test = dataset(data, transforms.Compose(transforms_test))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetSequentialSampler(test_idx)

    train_loader = DataLoader(
        dataset_train, batch_size=1, sampler=train_sampler, num_workers=1,
        drop_last=True, pin_memory=True
    ) #Here we also lower num_workers from 2->1
    # test_loader = DataLoader(
    #     dataset_test, batch_size=1, sampler=test_sampler, num_workers=2,
    #     drop_last=False, pin_memory=True
    # )It's the same here, the GPU can't run, so it can only run on the CPU
    test_loader = DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=2,
        drop_last=False, pin_memory=False
    )

    return train_loader, test_loader


def train(training_path, train_idx, test_idx):
    transforms_train = [
        random_hsv,
        random_affine,
        random_vertical_flip,
        random_horizontal_flip,
        random_cutout,
        pad_images_unet,
        to_torch_tensors,
    ]

    transforms_test = [
        pad_images_unet,
        to_torch_tensors,
    ]

    train_loader, test_loader = create_dataloaders(
        dataset=VesselsDataset,
        data=config.data,
        train_idx=list(train_idx),
        test_idx=list(test_idx),
        transforms_train=transforms_train,
        transforms_test=transforms_test
    )

    # cnnet = R2Vessels(
    #     base_channels=config.base_channels,
    #     in_channels=config.in_channels,
    #     out_channels=config.out_channels,
    #     num_iterations=config.num_iterations,
    #     gpu_id=config.gpu_id,
    #     model=config.model,
    #     criterion=config.criterion,
    #     base_criterion=config.base_criterion,
    #     learning_rate=config.learning_rate
    # )
    
    # cnnet.training(
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     scheduler_patience=sys.maxsize,
    #     stopping_patience=200,
    #     path_to_save=training_path,
    # )
    cnnet = R2Vessels(
        base_channels=config.base_channels,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_iterations=config.num_iterations,
        gpu_id=config.gpu_id,
        model=config.model,
        criterion=config.criterion,
        base_criterion=config.base_criterion,
        learning_rate=config.learning_rate
    )

    # Try loading generator_best.pth to continue training
    checkpoint_path = "D:\GAVE-main\Log\GAVE_trainv1_GAVENet_lr1e-04_RRLoss-BCE3Loss_bc64\generator_best.pth"
    # init_iter = 9826
    init_iter = 11551
    if os.path.exists(checkpoint_path):
        print(f"🔁 Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)
        cnnet.model.load_state_dict(state_dict)
        
        # # Try to read the last saved iteration count
        # best_loss_path = "D:\GAVE-main\Log\GAVE_trainv0_GAVENet_lr1e-04_RRLoss-BCE3Loss_bc64/best_loss.csv"
        # if os.path.exists(best_loss_path):
        #     with open(best_loss_path, 'r') as f:
        #         lines = f.readlines()
        #         if len(lines) > 1:
        #             last_iter = int(float(lines[-1].strip().split(',')[1]))
        #             init_iter = last_iter + 1
        #             print(f"🔢 Resuming from iteration {init_iter}")
    else:
        print("🆕 No checkpoint found, training from scratch.")

    # 🔧 开始训练
    cnnet.training(
        train_loader=train_loader,
        test_loader=test_loader,
        scheduler_patience=sys.maxsize,
        stopping_patience=200,
        path_to_save=training_path,
        init_iter=init_iter
    )


def split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def train_sets(sets):
    current = multiprocessing.current_process() 
    process_id = str(current.pid)

    set_ = sets 
    train_imgs = set_['training']
    val_imgs = set_['validation']

    generator_pth = config.model
    if config.model in ['RRWNet', 'RRWNetAll', 'RRUNet']:
        generator_pth += '_{}it'.format(config.num_iterations)
    pattern = '{}/{}_{}_{}_lr{:.0e}_{}_bc{}'
    criterion_str = config.criterion
    if config.base_criterion is not None:
        criterion_str += '-' + config.base_criterion
    train_path = pattern.format(
        config.training_folder,
        config.dataset,
        config.version,
        generator_pth,
        config.learning_rate,
        criterion_str,
        config.base_channels,
    )

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    shutil.copy2('./Model/model.py', train_path)
    
    args_dict = {k: v for k, v in vars(config.args).items() if not k.startswith('__')}
    with open(os.path.join(train_path, 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    current_instance = {
        'train_path': train_path,
        'train_imgs': train_imgs,
        'val_imgs': val_imgs,
    }

    instance = current_instance 
    print('-----------------Training Info--------------------')
    print("PID:", process_id)
    print("Training path:", instance['train_path'])
    print("Training: ",instance['train_imgs'])
    print("Validation: ", instance['val_imgs'])
    print('-----------------Start Training-------------------')
    train(
        instance['train_path'],
        instance['train_imgs'],
        instance['val_imgs']
    )


# def multi_train():
#     folds = get_folds(
#         config.images,
#         num_folds=config.num_folds, 
#     )  
#     sets = folds 

#     train_sets(sets) 
def multi_train():
    # ✅ Get a list of all image IDs
    all_imgs = config.images  # e.g., ['g_001', ..., 'g_050']

    # ✅ Method 1: Hard-coded partitioning (you can modify it as needed)
    # train_imgs = all_imgs[:25]
    # val_imgs = all_imgs[25:]

    # ✅ Method 2: Randomly split, different for each training (you can fix the seed to keep it reproducible)
    import random
    random.seed(config.seed)
    shuffled = all_imgs.copy()
    random.shuffle(shuffled)
    split_idx = int(0.5 * len(shuffled))  # 50% training, 50% validation
    train_imgs = shuffled[:split_idx]
    val_imgs = shuffled[split_idx:]

    # ✅ Construct a dictionary and pass it into train_sets (the format remains unchanged)
    sets = {
        'training': train_imgs,
        'validation': val_imgs,
    }

    train_sets(sets)


    


def fix_seeds():
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


if __name__ == "__main__":
    fix_seeds()
    multi_train()
