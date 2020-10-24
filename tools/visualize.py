import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def visualization(args):
    log_path = os.path.join(args.root_log, args.store_name)
    checkpoint_path = os.path.join(args.root_model, args.store_name)
    current_checkpoint_path = os.path.join(checkpoint_path, "ckpt.pth.tar")
    best_checkpoint_path = os.path.join(checkpoint_path, "ckpt.best.pth.tar")

    p_record_path = os.path.join(log_path, "p_record.txt")
    if os.path.isfile(p_record_path):
        # 不是第一次
        try:
            with open(p_record_path, 'r') as f:
                p_record = int(f.read())
                assert 0 <= p_record <= 100000, "p_record out of range"
                print("history record p: ", p_record)
        except (IOError, ValueError) as E:
            print(E)
            print("the p_record.txt file is suddenly missing")
            p_record = 0
    else:
        # 第一次
        p_record = 0
        print("init record p: ", p_record)


    # get data
    try:
        files = os.listdir(log_path)
        log_pattern = re.compile('(log)(-*\d+)(\.csv)')
        log_number = re.compile('-*\d+')
        log_numbers = []
        for f in files:
            if log_pattern.match(f):
                this_log_number = int(log_number.findall(f)[0])
                log_numbers.append(this_log_number)
        min_log_number = min(log_numbers)
        min_log = 'log' + str(min_log_number) + '.csv'
        data = pd.read_csv(os.path.join(log_path, min_log), delimiter='\n', engine='python', header=None,
                           error_bad_lines=False)
        data = [str(x) for x in data.values]
        test = [re.findall(r'-?\d+\.?\d*e?-?\d*?', x) for x in data if "Testing" in x]
        train = [re.findall(r'-?\d+\.?\d*e?-?\d*?', x) for x in data if "Worker" in x]
        y_val = [float(x[1]) for x in test]
        y_train_batch = [float(x[-5]) for x in train]
        y_train_avg_epoch = []

        loss_val = [float(x[-1]) for x in test]
        loss_train_batch = [float(x[9]) for x in train]
        CE_loss_train_batch = [float(x[11]) for x in train]
        KL_loss_train_batch = [float(x[13]) for x in train]
        loss_train_avg_epoch = []
        CE_loss_train_avg_epoch = []
        KL_loss_train_avg_epoch = []

        for i, tmp in enumerate(train):
            if float(tmp[-5]) == float(tmp[-4]) and i > 0:
                y_train_avg_epoch.append(float(train[i - 1][-4]))
                loss_train_avg_epoch.append(float(train[i - 1][10]))
                CE_loss_train_avg_epoch.append(float(train[i - 1][12]))
                KL_loss_train_avg_epoch.append(float(train[i - 1][14]))

        num_epochs = len(y_train_avg_epoch)
        num_batchs = len(y_train_batch)
        num_vals = len(y_val)
        x_train_avg_epoch = np.array(range(num_epochs))
        x_train_batch = np.array(range(num_batchs)) * (num_epochs / num_batchs)
        x_val = np.array(range(num_vals)) * (num_epochs / num_vals)

        best_val_y = max(y_val)
        best_val_x = x_val[y_val.index(best_val_y)]

        if True:
            current_checkpoint = torch.load(current_checkpoint_path, map_location=lambda storage, loc: storage)
            best_checkpoint = torch.load(best_checkpoint_path, map_location=lambda storage, loc: storage)
            current_state_dict = current_checkpoint['state_dict']
            best_state_dict = best_checkpoint['state_dict']
            assert current_state_dict.keys() == best_state_dict.keys()
            current_p_logit = [(x, torch.sigmoid(torch.tensor(float(current_state_dict[x])))) for x in
                               current_state_dict.keys() if
                               "p_logit" in x]
            best_p_logit = [(x, torch.sigmoid(torch.tensor(float(best_state_dict[x])))) for x in
                               best_state_dict.keys() if
                               "p_logit" in x]
            X_p = list(range(len(current_p_logit)))
            X_p_ticks = [loc for loc, value in current_p_logit]
            Y_p_current = np.array([value.item() for loc, value in current_p_logit])
            Y_p_best = np.array([value.item() for loc, value in best_p_logit])
    except Exception as e:
        print("visual exception: ", e)
        return

    else:
        print("log and checkpoint data load success, generating the result picture")
        # result
        plt.figure(figsize=(20, 10))

        plt.subplot(121)
        plt.title("prec1@{}".format("something"))
        plt.plot(x_train_batch, y_train_batch, label="train batchs")
        plt.plot(x_train_avg_epoch, y_train_avg_epoch, marker='*', label="train epochs average")
        plt.plot(x_val, y_val, marker='o', label="test per {} epochs".format(round(num_epochs / num_vals)))
        plt.annotate('best: {}'.format(best_val_y),
                     xy=(best_val_x, best_val_y),
                     xycoords='data',
                     xytext=(50, 50),
                     textcoords='offset points',
                     fontsize=16,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3, rad=.2"))

        plt.xlabel("epochs")
        plt.ylabel("top1%")
        plt.legend(loc="best")

        plt.subplot(122)
        plt.title("loss@{}".format("something"))
        plt.plot(x_train_batch, loss_train_batch, label="train batchs' loss")
        plt.plot(x_train_batch, CE_loss_train_batch, label="train batchs' CrossEntropy loss")
        plt.plot(x_train_batch, KL_loss_train_batch, label="train batchs' KL loss")
        plt.plot(x_train_avg_epoch, loss_train_avg_epoch, marker="*", label="train epoch avg's loss")
        plt.plot(x_train_avg_epoch, CE_loss_train_avg_epoch, marker="*", label="train epoch avg's CE loss")
        plt.plot(x_train_avg_epoch, KL_loss_train_avg_epoch, marker="*", label="train epoch avg's KL loss")
        plt.plot(x_val, loss_val, marker="*", label="test per {} epochs' loss".format(round(num_epochs / num_vals)))
        plt.xlabel("epochs")
        plt.ylabel("loss")

        plt.legend(loc='best')

        plt.savefig(os.path.join(log_path, "result.png"), bbox_inches='tight')

        if True:

            # current p
            p_record += 1
            with open(p_record_path, "w") as f:
                f.write(str(p_record))

            plt.figure(figsize=(20, 10))
            plt.bar(X_p, Y_p_current, facecolor='#ff9800')
            plt.plot(X_p, Y_p_current, marker='^', markersize=15, linewidth=3)
            plt.title("p to drop connection")
            # 显示数据
            for x, y in zip(X_p, Y_p_current):
                plt.text(x, y + 0.03,
                         '%.2f' % y, ha='center', va='bottom',
                         fontdict={'color': '#0091ea',
                                   'size': 16})
            plt.ylim(0., 1.)
            plt.xticks(X_p, X_p_ticks, size="small", rotation=85)
            plt.yticks([])
            plt.savefig(os.path.join(log_path, "current{}_p.png".format(p_record)), bbox_inches='tight')

            # best p
            plt.figure(figsize=(20, 10))
            plt.bar(X_p, Y_p_best, facecolor='#0000FF')
            plt.plot(X_p, Y_p_best, marker='^', markersize=15, linewidth=3)
            plt.title("p to drop connection")
            # 显示数据
            for x, y in zip(X_p, Y_p_best):
                plt.text(x, y + 0.03,
                         '%.2f' % y, ha='center', va='bottom',
                         fontdict={'color': '#0091ea',
                                   'size': 16})
            plt.ylim(0., 1.)
            plt.xticks(X_p, X_p_ticks, size="small", rotation=85)
            plt.yticks([])
            plt.savefig(os.path.join(log_path, "best_p.png"), bbox_inches='tight')

        plt.close('all')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch implementation of NAS_spatiotemporal")
    parser.add_argument('--root_log', type=str, default='/mnt/log/NAS_spatiotemporal/log')
    parser.add_argument('--root_model', type=str, default='/mnt/log/NAS_spatiotemporal/checkpoint')
    parser.add_argument('--store_name', type=str, default="")

    args = parser.parse_args()
    visualization(args)