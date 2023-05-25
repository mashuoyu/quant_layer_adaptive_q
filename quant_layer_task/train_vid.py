# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys
import math

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MeanScaleHyperprior,Adaptive_q_embedded,JointAutoregressiveHierarchicalPriors


from compressai.datasets import ImageFolder
from optimizer import net_aux_optimizer

from darknet import load_model

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target_f, f_hat, d):
        N, _, H, W = d.size()
        out = {}
        num_pixels = N * H * W
        logq=(torch.log(torch.tensor([1,2,3]))/math.log(2)).to("cuda")
        out["y_bpp_loss"] = torch.log(output["y1_likelihoods"]).sum() / (-math.log(2) * num_pixels)
        out["qloss"]=torch.dot(output["q1_f"],logq)
        out["info_loss"]=self.metric(output["mu_hat"]/torch.sqrt(2*output["scale"]), target_f/torch.sqrt(2*output["scale"]))+torch.sum(torch.sqrt(output["scale"]))
        out["f_loss"]=self.metric(f_hat, target_f)
        out["loss"] = out["y_bpp_loss"]+400*255*255*self.lmbda*out["info_loss"]-out["qloss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer


def freeze_model(model, to_freeze_dict, keep_step=None):

    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model


def train_one_epoch(
    adnet, model, model_yolo, criterion, train_dataloader, optimizer, epoch, clip_max_norm, device
):
    adnet.train()
    model.eval()
    model_yolo.eval()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()

        out_net_com = model.get_compression_vid(d,False)
        out_net=adnet(out_net_com["y_hat"],out_net_com["scales_hat"],out_net_com["means_hat"])
        out_f=model_yolo(d,"training")
        fsim=model_yolo.get_f13_sim(out_net["f_hat"])
        out_criterion = criterion(out_net,out_f,fsim,d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(adnet.parameters(), clip_max_norm)
        optimizer.step()

        q1_f=out_net["q1_f"][0].item()
        q2_f=out_net["q1_f"][1].item()
        q3_f=out_net["q1_f"][2].item()
        #q4_f=out_net["q1_f"][3].item()
        if i % 10 == 0:

            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tF13 loss: {out_criterion["f_loss"].item():.3f} |'
                f'\tInfo loss: {out_criterion["info_loss"].item():.3f} |'
                f"\tq_fre: {q1_f:.2f} {q2_f:.2f} {q3_f:.2f} |"
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.2f} |' 

            )





def test_epoch(epoch, test_dataloader, adnet, model, model_yolo, criterion, device):
    adnet.eval()
    model.eval()
    model_yolo.eval()

    loss = AverageMeter()
    y_bpp_loss = AverageMeter()

    f_loss = AverageMeter()
    q1_f= AverageMeter()
    q2_f= AverageMeter()
    q3_f= AverageMeter()
    info_loss=AverageMeter()
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net_com = model.get_compression_vid(d,True)
            out_net=adnet(out_net_com["y_hat"],out_net_com["scales_hat"],out_net_com["means_hat"])
            out_f=model_yolo(d,"training")
            q1_fx=out_net["q1_f"][0].item()
            q2_fx=out_net["q1_f"][1].item()
            q3_fx=out_net["q1_f"][2].item()
            fsim=model_yolo.get_f13_sim(out_net["f_hat"])
            out_criterion = criterion(out_net,out_f,fsim,d)

            y_bpp_loss.update(out_criterion["y_bpp_loss"])
            loss.update(out_criterion["loss"])
            f_loss.update(out_criterion["f_loss"])
            info_loss.update(out_criterion["info_loss"])
            q1_f.update(q1_fx)
            q2_f.update(q2_fx)
            q3_f.update(q3_fx)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tf13 loss: {f_loss.avg:.3f} |"
        f"\tinfo loss: {info_loss.avg:.3f} |"
        f"\tq_fre: {q1_f.avg:.2f} {q2_f.avg:.2f} {q3_f.avg:.2f} |"
        f"\ty_bpp_loss: {y_bpp_loss.avg:.3f} \n"

        
    )

    return loss.avg

def load_checkpoint(net, no_update: bool, checkpoint_path: str) -> nn.Module:
    # update model if need be
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    state_dict = checkpoint.copy() 
    for name in checkpoint.keys():
        for i in range(5):
            name_change1=f"entropy_bottleneck._matrices.{i:d}"
            name_change2=f"entropy_bottleneck._factors.{i:d}"
            name_change3=f"entropy_bottleneck._biases.{i:d}"
            if name==name_change1:
                state_dict[f"entropy_bottleneck._matrix{i:d}"] = state_dict.pop(name) 
            if name==name_change2:
                state_dict[f"entropy_bottleneck._factor{i:d}"] = state_dict.pop(name) 
            if name==name_change3:
                state_dict[f"entropy_bottleneck._bias{i:d}"] = state_dict.pop(name) 

    # compatibility with 'not updated yet' trained nets
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]

    net = net.from_state_dict(state_dict)
    if not no_update:
        net.update(force=True)
    return net.eval(),state_dict


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=400,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.006,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    net = JointAutoregressiveHierarchicalPriors(192,192)
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        last_epoch = 0
        net, state_dict=load_checkpoint(net,True,args.checkpoint)
        #optimizer.load_state_dict(checkpoint["optimizer"])
        #aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        #lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    adnet=Adaptive_q_embedded(192)
    yolo = load_model("/home/gujicheng/msy/PyTorch-YOLOv3/config/yolov3.cfg","/home/gujicheng/msy/PyTorch-YOLOv3/weights/yolov3.weights")
    yolo.to(device)
    adnet.to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer= configure_optimizers(adnet, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0


    net = net.to(device)
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            adnet,
            net,
            yolo,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            device,
        )
        loss = test_epoch(epoch, test_dataloader, adnet, net, yolo, criterion, device)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": adnet.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
