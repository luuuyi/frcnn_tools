import os

SRC_FILE_TEST = 'test_image_index.txt'
SRC_FILE_TRAIN = 'train_image_index.txt'

DST_FILE_TEST = 'test_image_index_modify.txt'
DST_FILE_TRAIN = 'train_image_index_modify.txt'

CHANGE = '/home/luyi/share/data_tingjiping/'

def change_root(read_file_name, write_file_name):
    with open(read_file_name, 'r') as fd:
        with open(write_file_name, 'a+') as fd_w:
            for line in fd.readlines():
                pos = line.find('air')
                new_line = line[pos:]
                new_line.replace('\\', '/')
                fd_w.write(new_line)
    print '%s Done...' % (read_file_name)

if __name__ == '__main__':
    change_root(SRC_FILE_TEST, DST_FILE_TEST)
    change_root(SRC_FILE_TRAIN, DST_FILE_TRAIN)