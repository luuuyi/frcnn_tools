# coding: utf-8

import os
import glob
import sys
import matplotlib.pyplot as plt
from time import sleep
import numpy as np

def calc_moving_average(data, moving_len):
    if len(data) < moving_len:
        return data
    rdata = data[:]
    s = sum(data[:moving_len])
    rdata[moving_len - 1] = s / moving_len
    for i in range(moving_len, len(data)):
        s = s - data[i - moving_len] + data[i]
        rdata[i] = s / moving_len
    return rdata
    
def calc_moving_average2(data, moving_len):
    s = 0
    rdata = data[:]
    for i in range(0, len(data)):
        if i < moving_len:
            s += data[i]
            d = i + 1
        else:            
            s = s - data[i - moving_len] + data[i]
            d = moving_len
        rdata[i] = s / d
    return rdata

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print 'usage: analyze_caffe_info.py info_filepath'
    else:
        while 1:
            train_iter = []
            train_loss = []
            train_acc_iter = []
            train_acc = []
            test_loss_iter = []
            test_loss = []
            test_acc_iter = []
            test_acc = []
            count = 0
            with open(sys.argv[1], 'r') as file:
                for line in file:
                    #if ('Train net output' in line) and ('loss' in line):
                    #    loss = float(line.split('=')[1].split('(')[0])
                    #    train_loss.append(loss)
                    if 'Iteration' in line:
                        iter = int(line.split('Iteration ')[1].split(',')[0])
                    if 'Iteration' in line and 'loss' in line:
                    # if ' rpn_loss_bbox =' in line:
                        loss = float(line.split('=')[1].strip().split(' ')[0])
                        train_loss.append(loss)
                        train_iter.append(iter)
            xmax = max(train_iter)
            train_loss_mv = calc_moving_average2(train_loss, 200)
                        
            min_loss = 1e10
            min_loss_pos = -1
            for i, l in zip(train_iter, train_loss_mv):
                print i, l
                if l < min_loss:
                    min_loss = l
                    min_loss_pos = i
            print 'min:', min_loss_pos, min_loss
            
            print "sorted last ten:"
            mv_arr = np.array(train_loss_mv)
            sort_inx = np.argsort(mv_arr)
            for i in range(min(20, sort_inx.size)):
                print train_iter[sort_inx[i]], mv_arr[sort_inx[i]]
            
            #plt.subplot(211)
            plt.plot(train_iter, train_loss, 'b', label='TrainLoss')            
            plt.plot(train_iter, train_loss_mv, 'r', label='TrainLossMovingAverage')
            plt.legend()
            plt.xlim(0, xmax)
            plt.ylim(0.0, 4.0)
            #plt.show(block=False)
            
            
            plt.xlabel(sys.argv[1])
            
            ax = plt.gca()
            ax.yaxis.set_ticks_position('right')
            
            plt.show(block=True)            
            
            #plt.savefig('%s.jpg' % os.path.basename(sys.argv[1]))
            
            break
            
