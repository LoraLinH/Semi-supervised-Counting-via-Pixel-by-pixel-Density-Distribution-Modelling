from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.model import vgg19_trans
from datasets.crowd_semi import Crowd
from losses.bay_loss import Bay_Loss
from losses.post import Post_Prob
from math import ceil


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    label = transposed_batch[4]
    return images, points, targets, st_sizes, label


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x, args.info) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = vgg19_trans()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device), strict=False)

        self.post_prob = Post_Prob(args.sigma, args.crop_size,
                                   args.downsample_ratio, args.background_ratio,
                                   args.use_background, self.device)

        self.criterion = Bay_Loss(args.use_background, self.device)
        self.criterion_mse = torch.nn.MSELoss(reduction='sum')
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_all = args.save_all
        self.best_count = 0
        idx_count2 = torch.tensor([0, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624, 0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833, 0.1263212381117904, 0.15925543689080027, 0.19863706203617743, 0.24597249461239232, 0.3025175130111165, 0.3707221162631514, 0.4537206813235279, 0.5560940547912038, 0.6838185522926952, 0.8476390438597705, 1.0642417040590761, 1.3645639664610938, 1.8055319029995607, 2.541316177212592, 3.87642023839676, 8.247815291086832])
        self.idx_count2 = idx_count2.unsqueeze(1).to(self.device)
        label_count2 = torch.tensor(
            [0.00016, 0.0048202634789049625, 0.01209819596260786, 0.02164922095835209, 0.03357841819524765, 0.04810526967048645, 0.06570728123188019, 0.08683456480503082, 0.11207923293113708, 0.1422334909439087, 0.17838051915168762, 0.22167329490184784, 0.2732916474342346, 0.33556100726127625, 0.41080838441848755, 0.5030269622802734, 0.6174761652946472, 0.762194037437439, 0.9506691694259644, 1.2056223154067993, 1.5706151723861694, 2.138580322265625, 3.233219861984253, 7.914860725402832])
        self.label_count2 = label_count2.unsqueeze(1).to(self.device)
        idx_count = torch.tensor(
            [0, 0.0008736941759623788, 0.00460105649110827, 0.011909992029514994, 0.021447560775165905, 0.03335742127399603, 0.04785158393927123, 0.06538952954794941, 0.08647975537451662, 0.11168024780931907, 0.14175821026385504, 0.17778540202168958, 0.22097960677712483, 0.2724192081348686, 0.3344926685808885, 0.40938709885499597, 0.5012436541947841, 0.6149288298909453, 0.7585325340575756, 0.9452185066011628, 1.1967563985336944, 1.5541906336372862, 2.0969205546489382, 2.9970217618726727, 4.51882041862729]) #25
        self.idx_count = idx_count.unsqueeze(1).to(self.device)
        label_count = torch.tensor(
            [0.00016, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624, 0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833, 0.1263212381117904, 0.15925543689080027, 0.19863706203617743, 0.24597249461239232, 0.3025175130111165, 0.3707221162631514, 0.4537206813235279, 0.5560940547912038, 0.6838185522926952, 0.8476390438597705, 1.0642417040590761, 1.3645639664610938, 1.8055319029995607, 2.541316177212592, 3.87642023839676]) #24
        self.label_count = label_count.unsqueeze(1).to(self.device)


    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch(epoch >= args.unlabel_start)
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self, unlabel):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, points, targets, st_sizes, label) in enumerate(self.dataloaders['train']):
            if not (unlabel | label[0]):
                continue
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                N = inputs.size(0)

                outputs, outputs2 = self.model(inputs)
                outputs = torch.softmax(outputs, dim=1)
                outputs2 = torch.softmax(outputs2, dim=1)
                cross_outputs = outputs.flatten(2)[0]
                cross_outputs2 = outputs2.flatten(2)[0]
                bay_outputs1 = torch.einsum("bixy,io->boxy", outputs, self.idx_count)
                bay_outputs2 = torch.einsum("bixy,io->boxy", outputs2, self.idx_count2)
                entro = torch.max(outputs, dim=1)[0]
                entro2 = torch.max(outputs2, dim=1)[0]
                mask = entro / (entro + entro2)
                mask = mask.unsqueeze(1).float()
                bay_outputs = (bay_outputs1 * mask + bay_outputs2 * (1 - mask))

                cross_outputs = cross_outputs.T
                cross_outputs2 = cross_outputs2.T

                if label[0]:
                    prob_list, gau = self.post_prob(points, st_sizes)
                    gaum = gau.flatten().unsqueeze(0) - self.label_count
                    gaum2 = gau.flatten().unsqueeze(0) - self.label_count2

                    gaum = torch.sum(gaum > 0, dim=0)
                    gaum2 = torch.sum(gaum2 > 0, dim=0)

                    gau = gaum.long()
                    gau2 = gaum2.long()
                    one_hot = torch.zeros_like(cross_outputs).scatter_(1, gau.unsqueeze(-1), 1)
                    one_hot = torch.cumsum(one_hot, dim=1)
                    cross_outputs = torch.cumsum(cross_outputs, dim=1)
                    loss = self.criterion_mse(one_hot, cross_outputs)
                    one_hot2 = torch.zeros_like(cross_outputs2).scatter_(1, gau2.unsqueeze(-1), 1)
                    one_hot2 = torch.cumsum(one_hot2, dim=1)
                    cross_outputs2 = torch.cumsum(cross_outputs2, dim=1)
                    loss += self.criterion_mse(one_hot2, cross_outputs2)


                else:
                    thresh = 0.5
                    mask1 = torch.max(cross_outputs, dim=1)[0] > thresh
                    mask2 = torch.max(cross_outputs2, dim=1)[0] > thresh
                    mask = (mask1 & mask2).detach()
                    loss = 0.1 * self.criterion_mse(bay_outputs1.flatten()[mask], bay_outputs2.flatten()[mask])

                epoch_loss.update(loss.item(), N)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



                pre_count = torch.sum(bay_outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count

                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            c_size = 3584
            if h >= c_size or w >= c_size:
                h_stride = int(ceil(1.0 * h / c_size))
                w_stride = int(ceil(1.0 * w / c_size))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output, output2 = self.model(input)
                        output = torch.softmax(output, dim=1)
                        output2 = torch.softmax(output2, dim=1)
                        bay_outputs = torch.einsum("bixy,io->boxy", output, self.idx_count)
                        bay_outputs2 = torch.einsum("bixy,io->boxy", output2, self.idx_count2)
                        # bay_outputs = (bay_outputs + bay_outputs2) / 2
                        entro = torch.max(output, dim=1)[0]
                        entro2 = torch.max(output2, dim=1)[0]
                        # mask = entro > entro2
                        mask = entro / (entro + entro2)
                        mask = mask.unsqueeze(1).float()
                        bay_outputs = (bay_outputs * mask + bay_outputs2 * (1 - mask))
                        pre_count += torch.sum(bay_outputs)
                res = count[0].item() - pre_count.item()
                epoch_res.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs, outputs2 = self.model(inputs)
                    outputs = torch.softmax(outputs, dim=1)
                    outputs2 = torch.softmax(outputs2, dim=1)
                    bay_outputs = torch.einsum("bixy,io->boxy", outputs, self.idx_count)
                    bay_outputs2 = torch.einsum("bixy,io->boxy", outputs2, self.idx_count2)
                    # bay_outputs = (bay_outputs + bay_outputs2) / 2
                    entro = torch.max(outputs, dim=1)[0]
                    entro2 = torch.max(outputs2, dim=1)[0]
                    # mask = entro > entro2
                    mask = entro / (entro + entro2)
                    mask = mask.unsqueeze(1).float()
                    bay_outputs = (bay_outputs * mask + bay_outputs2 * (1 - mask))
                    # save_results(inputs, outputs, self.vis_dir, '{}.jpg'.format(name[0]))
                    res = count[0].item() - torch.sum(bay_outputs).item()
                    epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            if self.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))



