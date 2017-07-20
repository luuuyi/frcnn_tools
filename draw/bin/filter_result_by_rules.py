#! usr/bin/python
#  encoding=utf-8

import os,glob,shutil

dst_file = u'.\\filter_detect_result\\'
obj_type_list = ["bus", "car", "person", "truck", "tricycle", "bicycle"]
filter_position_list = [u"00001_园区东北角快球_CVR_SEQNBR871_20150804", u"00009_重庆江北花园小区门口_20150522", u"00001_葛家镇潘家上口向南_20150521"]
filter_rect = [[0, 0, 680, 650],[230, 310, 370, 440], [738, 1, 822, 31]]
filter_type = [['bus', 'car', 'person', 'truck', 'tricycle'], ['car', 'tricycle'], ['bus', 'car', 'person', 'truck', 'tricycle']]
FILTER_CONF = 0.5
ignore_video_list = [u"00001_上海博物馆外国人_20150521", u"00002_上海博物馆中国人_20150521", u"00002_茶水间_20150522",
                     u"00001_园区大雪外景_20151022", u"00005_混行卡口场景三_20150521", u"00006_混行卡口场景三_20150521"]


def _compute_overlap_objs(det_rect, label_rect=None):
    # det_rect = det_rect.split(" ")
    # label_rect = label_rect.split(" ")
    # print det_rect[0], det_rect[1], det_rect[2], det_rect[3]

    det_x1, det_y1, det_x2, det_y2 = int(det_rect[0]), int(det_rect[1]), int(det_rect[2]), int(det_rect[3])
    label_x1, label_y1 = int(label_rect[0]), int(label_rect[1])
    label_x2, label_y2 = int(label_rect[2]), int(label_rect[3])
    common_rect = [max(det_x1, label_x1), max(det_y1, label_y1),
                   min(det_x2, label_x2), min(det_y2, label_y2)]
    over_w = int(common_rect[2]) - int(common_rect[0]) + 1
    over_h = int(common_rect[3]) - int(common_rect[1]) + 1
    if over_w > 0 and over_h > 0:
        common_area = over_w * over_h
        obj_area = (det_x2 - det_x1 + 1) * (det_y2 - det_y1 + 1)
        over_lap = float(common_area) / obj_area
        if over_lap > 0.5:
            return True
        else:
            return False
    else:
        return False


def _filter_str_context(obj_info, filter_rect):
    filter_str = []
    for i in xrange(len(obj_info)):
        if i < 2:
            filter_str.append(obj_info[i])
        else:
            rect = obj_info[i].split(" ")
            overlap = _compute_overlap_objs(rect[1:], filter_rect)
            if not overlap:
                filter_str.append(obj_info[i])
    filter_str[1] = str(len(filter_str) - 2)
    return " ".join(filter_str)


def _get_filter_str_line(str_line, filter_rect, filter_type):
    filter_str_line = []
    jpg_inx = str_line.find(".jpg")
    pic_name = str_line[0:jpg_inx + 4]
    filter_str_line.append(pic_name)
    for i in xrange(len(obj_type_list)):
        obj_info = []

        if obj_type_list[i] not in str_line:
            continue

        obj_inx = str_line.find(obj_type_list[i])
        str_left = str_line[obj_inx:].split(" ")
        obj_type = str_left[0]
        obj_num = str_left[1]
        obj_info.append(obj_type_list[i])
        obj_info.append(obj_num)
        i = 0
        while i < int(obj_num):
            temp = " ".join(str_left[2 + i * 5 + j] for j in xrange(5))
            obj_info.append(temp)
            i += 1
        if obj_info[0] in filter_type:
            obj_info_filter = _filter_str_context(obj_info, filter_rect)
        else:
            obj_info_filter = " ".join(obj_info)
        filter_str_line.append(obj_info_filter)
    return filter_str_line


def _filter_txt_position(txt, filter_position):
    print txt
    dst_txt = dst_file + txt.split(u"\\")[-1][:-4] + ".txt"
    for str_line in open(txt, 'r').readlines():
        if filter_position:
            for i in xrange(len(filter_position_list)):
                if filter_position_list[i] in str_line.decode("gb2312"):
                    filter_str_line = _get_filter_str_line(str_line, filter_rect[i], filter_type[i])
                    str_line = " ".join(filter_str_line)
        with open(dst_txt, "a+") as f:
            f.write(str_line.strip() + "\n")


def _init_label_dict():
    obj_dict = {}
    for obj_type in obj_type_list:
        obj_dict[obj_type] = []
    return obj_dict


def _single_obj_type_process(str_line, obj_type):
    obj_info_list = []
    if obj_type not in str_line:
        return obj_info_list
    inx = str_line.find(obj_type)
    str_list = str_line[inx:].split(" ")
    # print str_line, inx, obj_type
    obj_num = int(str_list[1])

    for i in xrange(obj_num):
        start_rect_inx = 2 + 5 * i
        conf = float(str_list[start_rect_inx])
        conf_cmp = FILTER_CONF
        if obj_type == u"person" or obj_type == u"car":
            conf_cmp += 0.15
        if conf >= conf_cmp:
            rect_info = str_list[start_rect_inx + 1:start_rect_inx + 5]
            rect_info.append(str_list[start_rect_inx])
            obj_info_list.append(rect_info)
    return obj_info_list


def _save_obj_type_result(pic_name, obj_dict, txt_name, save_file):
    for obj_type in obj_type_list:
        # if obj_type != u"person":
        #     break
        with open(save_file + obj_type + txt_name, "a+") as f:
            num_name = str(len(obj_dict[obj_type])) + " " + pic_name
            f.write(num_name)
            f.write("\n")
            for i in xrange(int(len(obj_dict[obj_type]))):
                rect_info = " ".join(obj_dict[obj_type][i])
                f.write(rect_info)
                f.write("\n")


def _save_merge_type_result(pic_name, obj_dict, txt_name):
    merge_type_num = 0
    merge_type_info_list = []
    for obj_type in merge_type_list:
        merge_type_num += int(len(obj_dict[obj_type]))
        for i in xrange(len(obj_dict[obj_type])):
            merge_type_info_list.append(obj_dict[obj_type][i])
    with open(save_file + txt_name, "a+") as f:
        num_name = str(merge_type_num) + " " + pic_name
        f.write(num_name)
        f.write("\n")
        for i in xrange(len(merge_type_info_list)):
            rect_info = " ".join(merge_type_info_list[i])
            f.write(rect_info)
            f.write("\n")


def _filter_result_by_conf(str_line, save_file):
    # print str_line
    str_line = str_line.strip()
    # print str_line
    pic_name = str_line[:str_line.find(".jpg") + len(".jpg")]
    # delete blank space
    # pic_name = "".join(pic_name.split())

    # pic_name = pic_name.split("/")[-1]
    obj_dict = _init_label_dict()

    for obj_type in obj_type_list:
        obj_dict[obj_type] = _single_obj_type_process(str_line.decode("gb2312"), obj_type)
    _save_obj_type_result(pic_name, obj_dict, "_total_result.txt", save_file)
    # save_merge_type_result(pic_name, obj_dict, "vehicle_total_result.txt")

    # print pic_name
    # video = pic_name[0:pic_name.rfind("/")].decode("gb2312")

    # b_ignore = False
    # for ignore_video in ignore_video_list:
    #     if video == ignore_video:
    #         b_ignore = True
    # if not b_ignore:
    #     txt_name = "_total_result" + str(62 - len(ignore_video_list)) + ".txt"
    #     save_obj_type_result(pic_name, obj_dict, txt_name)
    #     txt_name = "vehicle_total_result" + str(62 - len(ignore_video_list)) + ".txt"
    #     save_merge_type_result(pic_name, obj_dict, txt_name)


def filter_detect_result_by_rules(src_file=None, filter_position=True):
    if src_file is None:
        if os.path.exists(u".\\detect_txt\\"):
            src_file = u".\\detect_txt\\"
        else:
            print "Please input the detect_result path!"
            os.system("pause")

    if os.path.exists(dst_file):
        shutil.rmtree(dst_file)
    os.mkdir(dst_file)

    # filter result by position
    detect_result_txt_list = glob.glob(os.path.join(src_file, '*.txt'))
    for txt in detect_result_txt_list:
        _filter_txt_position(txt, filter_position)

    # filter result_by_conf
    txt_list = glob.glob(os.path.join(dst_file, "*txt"))
    for txt in txt_list:
        txt_name = txt.strip().split("\\")[-1][:-4]
        print txt_name
        save_file = dst_file + txt_name + "\/"
        os.makedirs(save_file)
        for str_line in open(txt, "r").readlines():
            _filter_result_by_conf(str_line, save_file)
        os.remove(txt)


if __name__ == '__main__':
    filter_detect_result_by_rules(None, True)  # default filter position is true




