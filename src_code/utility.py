import os
import math
import time
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


# -------------------------
# Timer Utility
# -------------------------
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0


# -------------------------
# Checkpoint Manager
# -------------------------
class checkpoint():
    def __init__(self, args):
        self.args = args
        self.log = torch.Tensor()
        self.ok = True

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # -------------------------
        # 👇 新保存路径：./Result/<save_name>
        # -------------------------
        save_root = './Result'
        save_name = args.save if args.save != '.' else now
        self.dir = os.path.join(save_root, save_name)

        # 创建基本目录
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'results'), exist_ok=True)

        # 日志文件
        log_file_path = os.path.join(self.dir, 'log.txt')
        self.log_file = open(log_file_path, 'a')

        # 保存 args
        with open(os.path.join(self.dir, 'config.txt'), 'a') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write(f'{arg}: {getattr(args, arg)}\n')
            f.write('\n')

    # -------------------------
    # Save model, optimizer, loss
    # -------------------------
    def save(self, trainer, epoch, is_best=False):
        # 保存模型
        trainer.model.save(self.dir, epoch, is_best=is_best)

        # 保存 loss
        trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)

        # 保存 PSNR 曲线
        # self.plot_psnr(epoch)

        # optimizer
        torch.save(trainer.optimizer.state_dict(),
                   os.path.join(self.dir, 'model', 'optimizer.pt'))

        # log
        torch.save(self.log, os.path.join(self.dir, 'model', 'psnr_log.pt'))

    # -------------------------
    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(os.path.join(self.dir, 'log.txt'), 'a')

    # -------------------------
    def done(self):
        self.log_file.close()

    # -------------------------
    # PSNR plot
    # -------------------------
    def plot_psnr(self, epoch):
        if len(self.log) == 0:
            return

        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title("PSNR Curve")

        plt.plot(axis, self.log[:, 0].numpy(), label='PSNR')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)

        plt.savefig(os.path.join(self.dir, 'psnr_curve.pdf'))
        plt.close(fig)

    # -------------------------
    # Save SR/LR/HR results（现用于去噪任务）
    # -------------------------
    def save_results_nopostfix(self, filename, save_list, scale):
        out_path = os.path.join(self.dir, 'results', filename)
        normalized = save_list[0][0].data.mul(255 / self.args.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        imageio.imsave(out_path + '.png', ndarr)


# -------------------------
# Quantize tensor to uint8
# -------------------------
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


# -------------------------
# PSNR calculator
# -------------------------
def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)

    # 不使用 SR benchmark shave，去噪任务只需 shave=scale=1
    shave = scale

    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


# -------------------------
# Optimizer
# -------------------------
def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    else:
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


# -------------------------
# Learning Rate Scheduler
# -------------------------
def make_scheduler(args, my_optimizer):
    return lrs.StepLR(
        my_optimizer,
        step_size=args.lr_decay,
        gamma=args.gamma
    )
