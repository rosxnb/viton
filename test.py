import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time

from tensorboardX import SummaryWriter

from datasets import CPDataset, CPDataLoader
from models.gmm import GMM
from models.unet import UnetGenerator
from utilities import load_checkpoint
from visualization import board_add_images, save_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default = False)

    opt = parser.parse_args()
    return opt


def test_gmm(opt, test_loader, model, board):
    if opt.use_cuda:
        model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)

    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)

    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']

        if opt.use_cuda:
            im = inputs['image'].cuda()
            im_pose = inputs['pose_image'].cuda()
            im_h = inputs['head'].cuda()
            shape = inputs['shape'].cuda()
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            im_c = inputs['parse_cloth'].cuda()
            im_g = inputs['grid_image'].cuda()
        else:
            im = inputs['image']
            im_pose = inputs['pose_image']
            im_h = inputs['head']
            shape = inputs['shape']
            agnostic = inputs['agnostic']
            c = inputs['cloth']
            cm = inputs['cloth_mask']
            im_c = inputs['parse_cloth']
            im_g = inputs['grid_image']

        # grid, theta = model(agnostic, c)
        grid, _ = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=False)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=False)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=False)
        # print(f"grid: {grid.shape}")
        # print(f"cm: {cm.shape}")
        # print(f"im_g: {im_g.shape}")
        # print(f"warped_cloth: {warped_cloth.shape}")
        # print(f"warped_mask: {warped_mask.shape}")
        # print(f"warped_grid: {warped_grid.shape}")

        visuals = [
            [im_h, shape, im_pose],
            [c, warped_cloth, im_c],
            [warped_grid, (warped_cloth + im) * 0.5, im]
        ]

        # print(f"    im_h size:          {im_h.shape}") # [4, 3, 256, 192]
        # print(f"    shape size:         {shape.shape}") # [4, 1, 256, 192]
        # print(f"    im_pose size:       {im_pose.shape}") # [4, 1, 256, 192]
        # print(f"    c size:             {c.shape}") # [4, 3, 256, 192]
        # print(f"    warped_cloth size:  {warped_cloth.shape}") # [4, 3, 256, 192]
        # print(f"    im_c size:          {im_c.shape}") # [4, 3, 256, 192]
        # print(f"    warped_grid size:   {warped_grid.shape}") # [4, 3, 256, 192]

        save_images(warped_cloth, c_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, c_names, warp_mask_dir)

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step + 1, t), flush=True)


def test_tom(opt, test_loader, model, board):
    if opt.use_cuda:
        model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)

    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)

    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im_names = inputs['im_name']
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        if opt.use_cuda:
            im = inputs['image'].cuda()
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
        else:
            im = inputs['image']
            agnostic = inputs['agnostic']
            c = inputs['cloth']
            cm = inputs['cloth_mask']



        outputs = model(torch.cat([agnostic, c], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [
            [im_h, shape, im_pose],
            [c, 2 * cm -1, m_composite],
            [p_rendered, p_tryon, im]
        ]

        # print(f"    im_h size:          {im_h.shape}") # [4, 3, 256, 192]
        # print(f"    shape size:         {shape.shape}") # [4, 1, 256, 192]
        # print(f"    im_pose size:       {im_pose.shape}") # [4, 1, 256, 192]
        # print(f"    c size:             {c.shape}") # [4, 3, 256, 192]
        # print(f"    cm size:            {cm.shape}") # [4, 1, 256, 192]
        # print(f"    m_composite size:   {m_composite.shape}") # [4, 1, 256, 192]
        # print(f"    p_rendered size:    {p_rendered.shape}") # [4, 3, 256, 192]
        # print(f"    p_tryon size:       {p_tryon.shape}") # [4, 3, 256, 192]
        # print(f"    im size:            {im.shape}") # [4, 3, 256, 192]

        save_images(p_tryon, im_names, try_on_dir)
        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step + 1, t), flush=True)


def main():
    opt = get_opt()
    print(opt)
    print(f'Start to test stage: {opt.stage}, named: {opt.name}!')

    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)

    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)

    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))

    # create model and train
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint, opt.use_cuda)

        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)

    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint, opt.use_cuda)

        with torch.no_grad():
            test_tom(opt, train_loader, model, board)

    else:
        raise NotImplementedError(f'Model [{opt.stage}] is not implemented')

    print(f'Finished test {opt.stage}, named: {opt.name}!')


if __name__ == '__main__':
    main()

