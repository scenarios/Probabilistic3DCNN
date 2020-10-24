import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of NAS_spatiotemporal")
parser.add_argument('--ckptpth', type=str, default='/mnt/log/NAS_spatiotemporal/checkpoint/NAS_sptp_nasas__ls250.0_something_RGB_Dense3D121_avg_segment1_e90_droprate0.0_num_dense_sample32_dense_sample_stride1_dense_Netv1d3Bz2by16Lr0.005SbnTpbtShare40-60-80/NAS_sptp_nasas_selection_ls250.0_something_RGB_Dense3D121_avg_segment1_e50_droprate0.0_num_dense_sample32_dense_Netv1d3Spbtsharels250temptSelect/ckpt.best.1.pth.tar')

import os
import numpy as np
import csv

def alpha_checker(state_dict, path=''):
    with open(os.path.join(path, 'p_log.csv'), mode='w') as csv_file:
        fields = ['index', 'S', 'T']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
        csv_writer.writeheader()

        records = {}
        for name, value in state_dict.items():
            if 'p_logit' in name and 'classifier' not in name:
                print('{} {}'.format(name, value.sigmoid().item()))
                name = name.replace('module.features.', '')
                name = name.replace('.conv1.p_logit', '')
                name = name.replace('.conv.p_logit', '')
                name = name.replace('bottleneck', 'S')
                name = name.replace('temporal', 'T')
                name = name.replace('denseblock', 'B')
                name = name.replace('denselayer', 'L')
                name = name.replace('original', 'S')
                if '.S' in name:
                    if name.replace('.S', '') in records.keys():
                        records[name.replace('.S', '')]['S'] = value.sigmoid().item()
                    else:
                        records[name.replace('.S', '')] = {}
                        records[name.replace('.S', '')]['S'] = value.sigmoid().item()
                    #csv_writer.writerow({'index': name, 'S': value.sigmoid().item(), 'T': state_dict[name.replace('.S', '.T')].sigmoid().item()})
                elif '.T' in name:
                    if name.replace('.T', '') in records.keys():
                        records[name.replace('.T', '')]['T'] = value.sigmoid().item()
                    else:
                        records[name.replace('.T', '')] = {}
                        records[name.replace('.T', '')]['T'] = value.sigmoid().item()
                else:
                    pass

        for name, value in records.items():
            csv_writer.writerow({'index': name, 'S': value['S'], 'T': value['T']})


def sptp_checker(state_dict):
    t_count = 0
    s_count = 0
    for name, value in state_dict.items():
        if 'p_logit' in name and 'classifier' not in name:
            #print('{}: {}'.format(name, value.sigmoid()))
            stensor = 1- np.floor(state_dict[name.replace('p_logit', 'unif_noise_variable')].cpu().item() + value.sigmoid().cpu().item())
            if int(stensor) == 0:
                if 'temporal' in name:
                    t_count += 1
                elif 'bottleneck' or 'original' in name:
                    s_count += 1
                print('{}: {}'.format(name, stensor))
        #if 'norm' in name:
        #    print('{}: {}'.format(name, value))
    print('{}: {}'.format('t_count', t_count))
    print('{}: {}'.format('s_count', s_count))


def main():
    args = parser.parse_args()
    import torch
    checkpoint = torch.load(os.path.join(args.ckptpth, 'ckpt.best.1.pth.tar'), map_location='cpu')
    state_dict = checkpoint['state_dict']
    #alpha_checker(state_dict, path=args.ckptpth)
    sptp_checker(state_dict)


if __name__ == '__main__':
    main()