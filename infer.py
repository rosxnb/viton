import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
from collections import namedtuple


import matplotlib.pyplot as plt

from models.gmm import GMM
from models.unet import UnetGenerator
from utilities import load_checkpoint
from datasets import CPDataset


def get_model_opt(use_cuda=False):
    options = namedtuple('options', ['fine_width', 'fine_height', 'radius', 'grid_size', 'use_cuda'])
    return options(fine_width=192, fine_height=256, radius=5, grid_size=5, use_cuda=use_cuda)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "test")
    parser.add_argument("--data_list", default = "test_pairs.txt")
    parser.add_argument("--result_dir", type=str, default="predictions", help="save inference result")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default = False)
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--stage", default = "GMM")

    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    model_opts = get_model_opt(use_cuda=opt.use_cuda)

    pretrained_gmm_path = os.path.join("checkpoints", "train_gmm_200K", "gmm_final.pth")
    pretrained_tom_path = os.path.join("checkpoints", "train_tom_200K", "tom_final.pth")

    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)

    data_loader = CPDataset(opt)

    while True:
        idx = int(input(f"choose an index (0, {len(data_loader)}]: "))
        inputs = data_loader[idx]

        _, ax = plt.subplots(4, 3, figsize=(15, 8), num="VITON Inference Results")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2)

        ax[0, 0].imshow( ( inputs['cloth'].detach().permute(1, 2, 0) * 0.5 ) + 0.5 )
        ax[0, 0].axis("off")
        ax[0, 0].set_title("Cloth")
        ax[0, 1].imshow( ( inputs['image'].detach().permute(1, 2, 0) * 0.5 ) + 0.5 )
        ax[0, 1].axis("off")
        ax[0, 1].set_title("Image")
        ax[0, 2].imshow( inputs["parse_image"] )
        ax[0, 2].axis("off")
        ax[0, 2].set_title("Parsed Image")

        ax[1, 0].imshow( ( inputs['head'].detach().permute(1, 2, 0) * 0.5 ) + 0.5 )
        ax[1, 0].axis("off")
        ax[1, 0].set_title("Head")
        ax[1, 1].imshow( ( inputs['shape'].detach().permute(1, 2, 0) * 0.5 ) + 0.5, cmap="gray" )
        ax[1, 1].axis("off")
        ax[1, 1].set_title("Shape")
        ax[1, 2].imshow( ( inputs["pose_image"].detach().permute(1, 2, 0) * 0.5 ) + 0.5, cmap="gray" )
        ax[1, 2].axis("off")
        ax[1, 2].set_title("Pose Keypoints")

        if opt.use_cuda:
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
            im_g = inputs['grid_image'].cuda()
        else:
            agnostic = inputs['agnostic']
            c = inputs['cloth']
            cm = inputs['cloth_mask']
            im_g = inputs['grid_image']

        im = inputs['image']

        # make batch=1
        agnostic.unsqueeze_(0)
        c.unsqueeze_(0)
        cm.unsqueeze_(0)
        im_g.unsqueeze_(0)

        # GMM predictions
        model = GMM(model_opts)
        load_checkpoint(model, pretrained_gmm_path, opt.use_cuda)

        with torch.no_grad():
            grid, _ = model(agnostic, c)

            warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=False)
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=False)
            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=False)

        ax[2, 0].imshow( ( warped_cloth.squeeze(0).cpu().detach().permute(1, 2, 0) * 0.5 ) + 0.5 )
        ax[2, 0].axis("off")
        ax[2, 0].set_title("Warp Cloth")
        ax[2, 1].imshow( ( warped_mask.squeeze(0).cpu().detach().permute(1, 2, 0) * 0.5 ) + 0.5, cmap="gray" )
        ax[2, 1].axis("off")
        ax[2, 1].set_title("Warp Mask")
        ax[2, 2].imshow( ( warped_grid.squeeze(0).cpu().detach().permute(1, 2, 0) * 0.5 ) + 0.5 )
        ax[2, 2].axis("off")
        ax[2, 2].set_title("Warp Grid")

        # TOM predictions
        model_tom = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model_tom, pretrained_tom_path, opt.use_cuda)

        with torch.no_grad():
            outputs = model_tom(torch.cat([agnostic, warped_cloth], 1))
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        ax[3, 0].imshow( ( (warped_cloth + im).squeeze(0).cpu().detach().permute(1, 2, 0) * 0.5 ) + 0.5 )
        ax[3, 0].axis("off")
        ax[3, 0].set_title("GMM Overlay")
        ax[3, 1].imshow( ( p_rendered.squeeze(0).cpu().detach().permute(1, 2, 0) * 0.5 ) + 0.5 )
        ax[3, 1].axis("off")
        ax[3, 1].set_title("Render Image")
        ax[3, 2].imshow( ( p_tryon.squeeze(0).cpu().detach().permute(1, 2, 0) * 0.5 ) + 0.5 )
        ax[3, 2].axis("off")
        ax[3, 2].set_title("Try On")

        plt.show()

        proceed = bool(input("want to continue? "))
        if not proceed: break


if __name__ == '__main__':
    main()


