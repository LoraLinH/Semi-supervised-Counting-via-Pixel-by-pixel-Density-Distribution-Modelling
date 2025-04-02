import torch
import os
import numpy as np
from datasets.crowd_semi import Crowd
from models.model import vgg19_trans
import argparse
import math
from glob import glob
from datetime import datetime

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/Dataset/Counting/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)

    model_list = sorted(glob(os.path.join(args.save_dir, '*.pth')))
    device = torch.device('cuda')
    model = vgg19_trans()
    model.to(device)
    model.eval()
    log_list = []
    # idx_count = torch.tensor(np.arange(0, 5, 0.1), dtype=torch.float32, device=device).unsqueeze(1)
    idx_count2 = torch.tensor([0, 0.001929451850323205, 0.008082773401606307, 0.016486622634959903, 0.027201606048777624,
                              0.040376651083361484, 0.05635653159451606, 0.07564311114549255, 0.09873047409540833,
                              0.1263212381117904, 0.15925543689080027, 0.19863706203617743, 0.24597249461239232,
                              0.3025175130111165, 0.3707221162631514, 0.4537206813235279, 0.5560940547912038,
                              0.6838185522926952, 0.8476390438597705, 1.0642417040590761, 1.3645639664610938,
                              1.8055319029995607, 2.541316177212592, 3.87642023839676, 8.247815291086832])
    idx_count2 = idx_count2.unsqueeze(1).to(device)
    idx_count = torch.tensor(
        [0, 0.0008736941759623788, 0.00460105649110827, 0.011909992029514994, 0.021447560775165905, 0.03335742127399603,
         0.04785158393927123, 0.06538952954794941, 0.08647975537451662, 0.11168024780931907, 0.14175821026385504,
         0.17778540202168958, 0.22097960677712483, 0.2724192081348686, 0.3344926685808885, 0.40938709885499597,
         0.5012436541947841, 0.6149288298909453, 0.7585325340575756, 0.9452185066011628, 1.1967563985336944,
         1.5541906336372862, 2.0969205546489382, 2.9970217618726727, 4.51882041862729])  # 25
    idx_count = idx_count.unsqueeze(1).to(device)

    for model_path in model_list:
        epoch_minus = []
        model.load_state_dict(torch.load(model_path, device))

        for inputs, count, name in dataloader:
            inputs = inputs.to(device)
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            c_size= 3584
            input_list = []
            if h >= c_size or w >= c_size:
                h_stride = int(math.ceil(1.0 * h / c_size))
                w_stride = int(math.ceil(1.0 * w / c_size))
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
                        outputs, outputs2 = model(input)
                        outputs = torch.softmax(outputs, dim=1)
                        outputs2 = torch.softmax(outputs2, dim=1)
                        bay_outputs = torch.einsum("bixy,io->boxy", outputs, idx_count)
                        bay_outputs2 = torch.einsum("bixy,io->boxy", outputs2, idx_count2)
                        entro = torch.max(outputs, dim=1)[0]
                        entro2 = torch.max(outputs2, dim=1)[0]
                        mask = entro / (entro + entro2)
                        mask = mask.unsqueeze(1).float()
                        bay_outputs = (bay_outputs * mask + bay_outputs2*(1-mask))
                        pre_count += torch.sum(bay_outputs)
                res = count[0].item() - pre_count.item()
                epoch_minus.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs, outputs2 = model(inputs)
                    outputs = torch.softmax(outputs, dim=1)
                    outputs2 = torch.softmax(outputs2, dim=1)
                    bay_outputs = torch.einsum("bixy,io->boxy", outputs, idx_count)
                    bay_outputs2 = torch.einsum("bixy,io->boxy", outputs2, idx_count2)
                    entro = torch.max(outputs, dim=1)[0]
                    entro2 = torch.max(outputs2, dim=1)[0]
                    mask = entro / (entro+entro2)
                    mask = mask.unsqueeze(1).float()
                    bay_outputs = (bay_outputs * mask + bay_outputs2*(1-mask))
                    pre_count = torch.sum(bay_outputs)
                    res = count[0].item() - pre_count.item()
                    epoch_minus.append(res)

        epoch_minus = np.array(epoch_minus)
        mse = np.sqrt(np.mean(np.square(epoch_minus)))
        mae = np.mean(np.abs(epoch_minus))
        log_str = 'model_name {}, mae {}, mse {}'.format(os.path.basename(model_path), mae, mse)
        log_list.append(log_str)
        print(log_str)

    date_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    with open(os.path.join(args.save_dir, 'test_results_{}.txt'.format(date_str)), 'w') as f:
        for log_str in log_list:
            f.write(log_str + '\n')
