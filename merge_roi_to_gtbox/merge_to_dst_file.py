import sys

SRC_FILE_1 = 'test_model_fanjing_1280.txt'
SRC_FILE_2 = 'test_roidb.txt'
DST_FILE   = 'person_total_result.txt'

def merge(input_1, input_2, output):
    with open(input_1, 'r') as fd1:
        with open(input_2, 'r') as fd2:
            with open(output, 'a+') as fd:
                pic_names = fd1.readlines()
                infos = fd2.readlines()
                for index, name in enumerate(pic_names):
                    line = []; infos_line = infos[index+1].split(' ')
                    line.append(name.split(' ')[0]); line.append(infos_line[0])
                    lens = len(infos_line[1:])/5
                    for i in xrange(lens):
                        start = 1 + i*5
                        cur_info = infos_line[start:start+4]
                        line.append(str(0)); line.extend(cur_info)
                    fd.write(' '.join(line)+'\n')

if __name__ == '__main__':
    merge(SRC_FILE_1, SRC_FILE_2, DST_FILE)