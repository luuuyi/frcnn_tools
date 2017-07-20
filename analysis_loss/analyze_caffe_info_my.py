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
        print ('usage: analyze_caffe_info.py info_filepath')
    else:
        while 1:

            train_iter1 = []
            train_loss1 = []
            train_cou_loss = []
            train_cou_iter = []
            
            
            train_acc_iter = []
            train_acc = []
            
            test_loss_iter = []
            test_cou_iter = []
            test_cou_loss = []
            test_den_iter = []
            test_den_loss = []
            test_blob_loss = []
            test_blob_iter = []
            
            test_acc_iter = []
            test_acc = []

            count = 0
            iters = 0

            train_iter = []
            train_loss = []

            test_iter =[]
            test_loss =[]

            with open(sys.argv[1], 'r') as file:
                for line in file:
                    #print ('%s\n', line)
                    if 'Iteration' in line:
                        if 'iter/s' in line:
                            iters = int(line.split('Iteration ')[1].split( )[0])
                        else:
                            iters = int(line.split('Iteration ')[1].split(',')[0])

                    #train_error:Iteration 18850, loss = 0.00462692
                    if 'Iteration' in line and 'loss' in line:
                        loss = float(line.split('=')[1].strip())
                        train_loss.append(loss)
                        train_iter.append(iters)
						
					#I0711 20:56:51.067550 24592 solver.cpp:730]     Test net output #0: top1/acc = 0.963684
                    if 'Test net output' in line and 'top1/acc' in line:
                        loss = float(line.split('=')[1].strip())
                        test_cou_loss.append(loss)
                        test_cou_iter.append(iters)	

            xmax = max(test_cou_iter)
            train_loss_mv = calc_moving_average2(train_loss, 200)
            train_loss1_mv = calc_moving_average2(train_loss1, 200)
            
            test_cou_loss_mv = calc_moving_average(test_cou_loss, 800)
            test_den_loss_mv = calc_moving_average(test_den_loss, 800)
                        
            max_accuracy = 0.01
            max_accuracy_pos = -1
            for i, l in zip(test_cou_iter, test_cou_loss):
                #print (i, l)
                if l > max_accuracy:
                    max_accuracy = l
                    max_accuracy_pos = i
            print ('min:', max_accuracy_pos, max_accuracy)
            
            #print ("sorted last ten:")
            #mv_arr = np.array(train_loss_mv)
            #sort_inx = np.argsort(mv_arr)
            #for i in range(min(20, sort_inx.size)):
            #    print (train_iter[sort_inx[i]], mv_arr[sort_inx[i]])
            

            plt.plot(train_iter, train_loss, 'b', label='TrainLoss')            
            plt.plot(train_iter, train_loss_mv, 'b--', label='TrainLossMovingAverage')
            plt.plot(test_cou_iter, test_cou_loss, 'r', label='TestCouLoss')
            plt.plot(test_den_iter, test_den_loss, 'b', label='TestDenLoss')

            
            plt.legend()
            plt.xlim(0, xmax)
            plt.ylim(0.0, 0.5)
            plt.show(block=False)
            
            
            plt.xlabel(sys.argv[1])
            
            ax = plt.gca()
            ax.yaxis.set_ticks_position('right')
            
            plt.show(block=True)            
            
            #plt.savefig('%s.jpg' % os.path.basename(sys.argv[1]))
            
            break
            
