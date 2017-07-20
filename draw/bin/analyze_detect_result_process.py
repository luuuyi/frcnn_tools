#! usr/bin/python
# encoding=utf8

import os, glob, shutil, sys
import numpy as np
import matplotlib.pyplot as plt

SCENE_PRERECALL = False

dst_file = u"./analyze_result/dst_recall_pre_result/"
hms_video_type = u"./bin/HMS_Video_Type.txt"
g_scene_csv_file = ""
g_video_csv_file = ""
g_fp_pic_info_csv_file = ""
g_all_type_name = "all_type"
g_all_obj_type_list = ["back_ground", "bus", "car", "person", "truck", "tricycle", "bicycle"]
g_merge_type_list = ["bus", "car", "truck"]
g_merge_type_name = "vehicle"
g_model_name = ""
g_gt_obj_size_list = []
OVERLAP = 0.5
SELF_NEG_OVERLAP = 0.3
g_conf_list = [0.75, 0.80, 0.95]
g_label_size_category = [20, 30, 40, 50, 60, 70, 80, 90, 100, 300,400,500,600,700]  # according to width
g_fppi_category = []
g_fppi_category_vehicle = [0.05, 0.1, 0.15, 0.2, 0.3]
g_fppi_category_person = [0.05, 0.1, 0.3, 0.4]
g_scene_dict = {}

g_color_turple = ('red', 'blue', 'black', 'fuchsia', 'lime', 'blueviolet', 'darkgreen', 'orange', 'gold', 'chartreuse')
g_curve_turple = ('fppi_fppi_recall', 'iou_iou_recall', 'score_fppi_conf', 'conf_conf_recall')


def init_csv_file(file_path):
    global g_all_obj_type_list

    global g_scene_csv_file
    g_scene_csv_file = file_path + "scene_pre_recall.csv"
    with open(g_scene_csv_file, "a+") as f:
        save_str = u"模型, 置信度, 置信度大小, 目标大小, 场景类型, Gt_num, Det_num, 正检数目, 误检数目, recall, precision\n"
        f.write(save_str.encode("gb2312"))

    global g_video_csv_file
    g_video_csv_file = file_path + "video_pre_recall.csv"
    with open(g_video_csv_file, "a+") as f:
        save_str = u"模型, 置信度, 置信度大小, 目标大小, 视频来源, 视频名称, Gt_num, Det_num, 正检数目, 误检数目, recall, precision, video_fppi"
        f.write(save_str.encode("gb2312"))
        save_str = ""
        for type in g_all_obj_type_list:
            save_str += ",fp_" + type + "_ratio"
        for type in g_all_obj_type_list:
            save_str += ",fp_" + type + "_num"
        f.write(save_str.encode("gb2312"))
        f.write("\n")

    global g_fp_pic_info_csv_file
    g_fp_pic_info_csv_file = file_path + "fp_pic_info.csv"
    with open(g_fp_pic_info_csv_file, "a+") as f:
        save_str = u"模型, 视频名称, 视频帧号, 错误数目\n"
        f.write(save_str.encode("gb2312"))


def init_video_type_dict():
    if not SCENE_PRERECALL:
        return
    global g_scene_dict
    if not os.path.exists(hms_video_type):
        print "Can't find ./HMS_Video_Type.txt!"
        exit(0)
    for str_line in open(hms_video_type, "r").readlines():
        str_line = str_line.strip().split(" ")
        if not g_scene_dict.has_key(str_line[0]):
            g_scene_dict[str_line[0]] = []
        g_scene_dict[str_line[0]].append(str_line[1])
    return g_scene_dict


def get_label_dict_per_type(file_path):
    label_result_dict = {}
    for str_line in open(file_path, "r").readlines():
        str_line_list = str_line.strip().split(" ")
        pic_name = str_line_list[0]
        obj_num = int(str_line_list[1])
        label_result_dict[pic_name] = []
        if 0 < obj_num:
            for i in xrange(obj_num):
                start = 3 + 5 * i
                obj_info = " ".join(str_line_list[start:start + 4])
                label_result_dict[pic_name].append(obj_info)
    return label_result_dict


def get_label_dict_all_type(label_txt_list):
    """get all type label dict

    Parameters:
        label_txt_list (list): all label file path
    Returns:
        label_dict_all_type (dict): a dict contain all label info
    """

    label_dict_all_type = {}
    for i in xrange(len(label_txt_list)):
        label_path = label_txt_list[i].replace("\\", '/')
        obj_type = label_path[label_path.rfind('/') + 1:].split("_")[0]
        label_dict_all_type[obj_type] = get_label_dict_per_type(label_path)
    return label_dict_all_type


def get_detect_result(txt_file):
    detect_result_dict = {}
    total_det_pic_num = 0
    pic_name = ""
    for str_line in open(txt_file, "r").readlines():
        # str_line = str_line.decode("utf-8").encode("gb2312")
        str_line_temp = str_line
        if ".jpg" in str_line:
            pic_name = str_line.strip().split(" ")[-1]
            detect_result_dict[pic_name] = []
            total_det_pic_num += 1
        elif (2 == len(str_line.strip().split(" "))):
            pic_name = str_line.strip().split(" ")[-1] + ".jpg"
            # print pic_name
            detect_result_dict[pic_name] = []
            total_det_pic_num += 1
        else:
            # print str_line.decode("utf8")
            detect_result_dict[pic_name].append(str_line.strip())
    return detect_result_dict


def get_match_label_detect_dict(label_dict_all_type, detect_result_dict, obj_type):

    # delete img no label info
    detect_keys = detect_result_dict.keys()
    label_keys = label_dict_all_type[obj_type].keys()
    #print detect_keys
    #print label_keys
    for key in detect_keys:
        if key not in label_keys:
            del detect_result_dict[key]

    # delete img not detect
    detect_keys = detect_result_dict.keys()
    label_keys = label_dict_all_type[obj_type].keys()
    for key in label_keys:
        if key not in detect_keys:
            for type in label_dict_all_type.keys():
                del label_dict_all_type[type][key]


def compute_overlap_objs(det_rect, label_rect):
    det_rect = det_rect.split(" ")
    label_rect = label_rect.split(" ")

    det_x1, det_y1, det_x2, det_y2 = int(det_rect[0]), int(det_rect[1]), int(det_rect[2]), int(det_rect[3])
    label_x1, label_y1 = int(label_rect[0]), int(label_rect[1])
    label_x2, label_y2 = int(label_rect[2]), int(label_rect[3])
    common_rect = [max(det_x1, label_x1), max(det_y1, label_y1),
                   min(det_x2, label_x2), min(det_y2, label_y2)]
    over_w = int(common_rect[2]) - int(common_rect[0]) + 1
    over_h = int(common_rect[3]) - int(common_rect[1]) + 1
    if over_w > 0 and over_h > 0:
        common_area = over_w * over_h
        merge_area = (det_x2 - det_x1 + 1) * (det_y2 - det_y1 + 1) +\
                     (label_x2 - label_x1 + 1) * (label_y2 - label_y1 + 1) - common_area
        over_lap = float(common_area) / merge_area
        return over_lap
    else:
        return 0


def get_overlap_objs(detect_obj, label_dict, obj_type):
    max_overlap = 0
    max_inx = -1
    all_max_overlap = 0
    all_max_inx = -1

    global g_merge_type_list
    global g_merge_type_name
    all_max_type = "back_ground"
    max_type = "back_ground"

    for key in label_dict.keys():
        label_per_type = label_dict[key]
        key = str(key)

        # if key != obj_type:
        #     continue

        if key == g_merge_type_name and obj_type != g_merge_type_name: #vehicle only useful when obj_type is vehicle
            continue
        if obj_type == g_merge_type_name and key in g_merge_type_list:  #vehicle not compare with merge_type_list
            continue

        for i in xrange(len(label_per_type)):
            overlap = compute_overlap_objs(detect_obj, label_per_type[i])
            if overlap > all_max_overlap:
                all_max_overlap = overlap
                all_max_inx = i
                all_max_type = key
            if overlap > max_overlap and key == obj_type:
                max_overlap = overlap
                max_inx = i
                max_type = key
    if max_overlap >= OVERLAP:
        return max_overlap, max_inx, max_type     
    else:
        return all_max_overlap, all_max_inx, all_max_type


def compute_per_obj_info(detect_list, label_dict_per_img, obj_type):
    match_dict = {}
    pos_list = []
    neg_list = []

    for i in xrange(len(detect_list)):
        max_overlap, max_inx, match_type = get_overlap_objs(detect_list[i], label_dict_per_img, obj_type)
        
        # pos(contain repeat)
        if (max_overlap >= OVERLAP) and match_type == obj_type:
            if match_type not in match_dict.keys():
                match_dict[match_type] = []
            if max_inx not in match_dict[match_type]:
                obj_info = label_dict_per_img[match_type][max_inx] + " " + detect_list[i] + " " + str(max_overlap)
                match_dict[match_type].append(max_inx)
                pos_list.append(obj_info)
            else:
                # delete repeat
                neg_info = detect_list[i] + " " + match_type
                neg_list.append(neg_info)
        else:
            # self_neg, others_neg, back_ground
            if max_overlap >= SELF_NEG_OVERLAP and match_type == obj_type:
                match_type = obj_type                 # [0.3, 0.5) self_neg
            elif max_overlap >= OVERLAP and match_type != obj_type:             
                match_type = match_type               # [0.5, ++) others neg
            else:
                match_type = "back_ground"
            neg_info = detect_list[i] + " " + match_type
            neg_list.append(neg_info)
    return pos_list, neg_list


def get_all_label_dict_img(all_label_dict, obj_type_process):
    all_label_dict_img = {}
    obj_types = all_label_dict.keys()

    for img_name in all_label_dict[obj_type_process].keys():
        img_info = {}
        for obj_type in obj_types:
            img_info[obj_type] = []
            try:
                img_info[obj_type] = all_label_dict[obj_type][img_name]  #in case of no this img_name
            except:
                img_info[obj_type] = []
        try:
            all_label_dict_img[img_name] = img_info
        except:
            all_label_dict_img[img_name] = []
            all_label_dict_img[img_name] = img_info
    return all_label_dict_img


def get_match_result(all_label_dict, detect_dict, obj_type):
    """
    match_result_dict:for per img: label_rect + detect_rect + conf + overlap
    no_match_result_dict: for per img: detect_rect + conf + no_match_type
    """

    match_dict = {}
    no_match_dict = {}
    all_label_dict_img = get_all_label_dict_img(all_label_dict, obj_type)
    
    for i in xrange(len(all_label_dict_img)):
        img_name = all_label_dict_img.keys()[i]
        try:         # in cafe of no this img
            detect_result_per_img = detect_dict[img_name]
        except:
            print img_name, "no this img"
            continue

        pos_list, neg_list = compute_per_obj_info(detect_result_per_img, all_label_dict_img[img_name], obj_type)
        match_dict[img_name] = pos_list
        no_match_dict[img_name] = neg_list
    return match_dict, no_match_dict


def get_fppi_conf_list_by_fppi(no_match_dict, pic_num, g_fppi_category):

    neg_conf_list = []
    conf_list = []

    # no_match_cnt = 0

    for value_list in no_match_dict.values():
        for value in value_list:
            # print value
            conf = float(value.split(" ")[-2])
            # print conf
            neg_conf_list.append(conf)
            # no_match_cnt += 1
    # print no_match_cnt

    neg_conf_list = sorted(neg_conf_list, reverse=True)
    for value in g_fppi_category:
        inx = min(int(value * pic_num), len(neg_conf_list) - 1)
        print inx, len(neg_conf_list), value, pic_num
        conf_list.append(neg_conf_list[inx])
    return conf_list


def get_gt_obj_size_list():
    global g_label_size_category
    obj_size_list = []
    for i in xrange(len(g_label_size_category)):
        if i == 0:
            obj_size_list.append(u"全部目标")
            obj_size_list.append(u"0--" + str(g_label_size_category[0]))
        else:
            size_name = str(g_label_size_category[i - 1]) + u"--" + str(g_label_size_category[i])
            obj_size_list.append(size_name)
    size_name = u"大于" + str(g_label_size_category[i])
    obj_size_list.append(size_name)
    return obj_size_list


def get_tp_gt_num_process(result_dict, min_size, max_size, conf_value=-1):
    tp_gt_num_dict = {}
    for pic_name in result_dict.keys():
        if "/" not in pic_name:
            inx = pic_name.rfind("_")
            video_name = "HMS_dataset/" + pic_name[:inx]
        else:
            video_name = "/".join(pic_name.split("/")[:-1])
        
        if not tp_gt_num_dict.has_key(video_name):
            tp_gt_num_dict[video_name] = 0
        obj_list = result_dict[pic_name]
        for obj in obj_list:
            rect = obj.split(" ")

            # 根据conf，判断是提取tp_num还是gt_num
            if conf_value != -1:
                start_inx = 0
                if float(rect[8]) < conf_value:
                    continue
            else:
                start_inx = 0
            x1, y1 = int(rect[start_inx]), int(rect[start_inx + 1])
            x2, y2 = int(rect[start_inx + 2]), int(rect[start_inx + 3])
            if min_size < x2 - x1 <= max_size:
                tp_gt_num_dict[video_name] += 1
    return tp_gt_num_dict


def get_fp_num_process(result_dict, min_size, max_size, conf_value):

    global g_all_obj_type_list, g_all_type_name, g_merge_type_name
    tp_gt_num_dict = {}
    for pic_name in result_dict.keys():
        if "/" not in pic_name:
            inx = pic_name.rfind("_")
            video_name = "HMS_dataset/" + pic_name[:inx]
        else:
            video_name = "/".join(pic_name.split("/")[:-1])

        if video_name not in tp_gt_num_dict.keys():
            tp_gt_num_dict[video_name] = {}
            for type in g_all_obj_type_list:
                tp_gt_num_dict[video_name][type] = 0
            tp_gt_num_dict[video_name][g_all_type_name] = 0
            tp_gt_num_dict[video_name][g_merge_type_name] = 0

        obj_list = result_dict[pic_name]
        for obj in obj_list:
            obj_info = obj.split(" ")
            fp_type = obj_info[5]
            start_inx = 0
            if conf_value <= float(obj_info[4]):
                x1, y1 = int(obj_info[start_inx]), int(obj_info[start_inx + 1])
                x2, y2 = int(obj_info[start_inx + 2]), int(obj_info[start_inx + 3])
                if min_size < x2 - x1 <= max_size:
                    tp_gt_num_dict[video_name][fp_type] += 1
                    tp_gt_num_dict[video_name][g_all_type_name] += 1
    return tp_gt_num_dict


def get_img_num_process(result_dict):
    img_num_dict = {}
    for pic_name in result_dict.keys():
        if "/" not in pic_name:
            inx = pic_name.rfind("_")
            video_name = "HMS_dataset/" + pic_name[:inx]
        else:
            video_name = "/".join(pic_name.split("/")[:-1])
        
        if not img_num_dict.has_key(video_name):
            img_num_dict[video_name] = 0
        img_num_dict[video_name] += 1
    return img_num_dict


def video_pre_recall_process(match_result_dict, no_match_result_dict, label_result_dict, conf, inx):
    video_pre_recall_list = []

    if inx == 0:              # 全部目标
        min_size = 0
        max_size = 10000
    elif inx == 1:            #
        min_size = 0
        max_size = g_label_size_category[inx - 1]
    elif inx == len(g_label_size_category) + 1:
        min_size = g_label_size_category[inx - 2]
        max_size = 10000
    else:
        min_size = g_label_size_category[inx - 2]
        max_size = g_label_size_category[inx - 1]
    # video_name, gt_num, tp_num, fp_num
    gt_num_dict = get_tp_gt_num_process(label_result_dict, min_size, max_size)
    tp_num_dict = get_tp_gt_num_process(match_result_dict, min_size, max_size, conf)
    fp_num_dict = get_fp_num_process(no_match_result_dict, min_size, max_size, conf)
    img_num_dict = get_img_num_process(label_result_dict)
    video_pre_recall_list.append(gt_num_dict)
    video_pre_recall_list.append(tp_num_dict)
    video_pre_recall_list.append(fp_num_dict)
    video_pre_recall_list.append(img_num_dict)
    return video_pre_recall_list


def scene_pre_recall_process(video_pre_recall_list):
    global g_scene_dict
    scene_pre_recall_dict = {}

    for scene in g_scene_dict.keys():
        gt_num, tp_num, fp_num = 0, 0, 0
        for video in g_scene_dict[scene]:
            gt_num += video_pre_recall_list[0][video]
            tp_num += video_pre_recall_list[1][video]
            for fp_type in video_pre_recall_list[2][video].keys():
                fp_num += video_pre_recall_list[2][video][fp_type]
        scene_pre_recall_dict[scene] = [gt_num, tp_num, fp_num]
    return scene_pre_recall_dict


def save_video_pre_recall_to_csv(video_pre_recall_list, conf_name, inx):
    global g_model_name, g_video_csv_file, g_all_obj_type_list, g_all_type_name
    with open(g_video_csv_file, "a+") as f:
        all_video_gt_num, all_video_tp_num, all_video_img_num = 0, 0, 0
        all_video_fp_num_type = {}
        for type in g_all_obj_type_list:
            all_video_fp_num_type[type] = 0
        all_video_fp_num_type[g_all_type_name] = 0
        all_video_fp_num_type[g_merge_type_name] = 0

        for video in video_pre_recall_list[0].keys():
            # print video
            # print video_pre_recall_list[0][video], video_pre_recall_list[1][video], video_pre_recall_list[3][video]

            gt_num, tp_num, img_num = video_pre_recall_list[0][video], video_pre_recall_list[1][video],\
                                      video_pre_recall_list[3][video]

            fp_num_type = {}
            for type in g_all_obj_type_list:
                fp_num_type[type] = 0
            fp_num_type[g_all_type_name] = 0

            for key in video_pre_recall_list[2][video].keys():
                fp_num_type[key] = video_pre_recall_list[2][video][key]
                all_video_fp_num_type[key] += fp_num_type[key]

            # 全部视频数据统计
            # 因为不同大小的误检不好区分，故不同目标大小的误检数目都相同
            all_video_gt_num += gt_num
            all_video_tp_num += tp_num
            all_video_img_num += img_num

            save_str = str(g_model_name) + "," + str(conf_name.split(" ")[-1]) + "," + str(conf_name.split(" ")[0]) + "," +\
                       str(g_gt_obj_size_list[inx].encode("gb2312")) + "," + str(video.split("/")[0]) + "," + str(video.split("/")[1]) + ","
            save_str += str(gt_num) + "," + str(tp_num + fp_num_type[g_all_type_name]) + "," + \
                        str(tp_num) + "," + str(fp_num_type[g_all_type_name]) + "," + \
                        str(float(tp_num) / max(0.00001, gt_num)) + "," + \
                        str(float(tp_num) / max(0.00001, fp_num_type[g_all_type_name] + tp_num)) + "," + \
                        str(float(fp_num_type[g_all_type_name]) / max(0.00001, img_num))
            for type in g_all_obj_type_list:
                save_str += "," + str(float(fp_num_type[type]) / max(0.00001, fp_num_type[g_all_type_name]))
            for type in g_all_obj_type_list:
                save_str += "," + str(fp_num_type[type])
            save_str += "\n"
            f.write(save_str)

        save_str = str(g_model_name) + "," + str(conf_name.split(" ")[-1]) + "," + str(conf_name.split(" ")[0]) + "," + \
                   str(g_gt_obj_size_list[inx].encode("gb2312")) + "," + "all_video" + "," + "all_video" + ","
        save_str += str(all_video_gt_num) + "," + str(all_video_tp_num + all_video_fp_num_type[g_all_type_name]) + "," + \
                    str(all_video_tp_num) + "," + str(all_video_fp_num_type[g_all_type_name]) + "," + \
                    str(float(all_video_tp_num) / max(0.00001, all_video_gt_num)) + "," + \
                    str(float(all_video_tp_num) / max(0.00001, all_video_fp_num_type[g_all_type_name] + all_video_tp_num)) + "," + \
                    str(float(all_video_fp_num_type[g_all_type_name]) / max(0.00001, all_video_img_num))
        for type in g_all_obj_type_list:
            save_str += "," + str(float(all_video_fp_num_type[type]) / max(1, all_video_fp_num_type[g_all_type_name]))
        for type in g_all_obj_type_list:
            save_str += "," + str(all_video_fp_num_type[type])
        save_str += "\n"
        f.write(save_str)


def save_scene_pre_recall_to_csv(scene_pre_recall_dict, conf_name, inx):
    global g_model_name, g_scene_csv_file
    with open(g_scene_csv_file, "a+") as f:
        total_gt_num, total_tp_num, total_fp_num = 0, 0, 0
        for scene in scene_pre_recall_dict.keys():
            gt_num, tp_num, fp_num = scene_pre_recall_dict[scene][0], scene_pre_recall_dict[scene][1], \
                                     scene_pre_recall_dict[scene][2]

            total_gt_num += gt_num
            total_tp_num += tp_num
            total_fp_num += fp_num

            save_str = str(g_model_name) + "," + str(conf_name.split(" ")[-1]) + "," + str(conf_name.split(" ")[0]) + "," + \
                       str(g_gt_obj_size_list[inx].encode("gb2312")) + "," + str(scene) + ","
            save_str += str(gt_num) + "," + str(tp_num + fp_num) + "," + \
                        str(tp_num) + "," + str(fp_num) + "," + \
                        str(float(tp_num) / max(0.00001, gt_num)) + "," + \
                        str(float(tp_num) / max(0.00001, fp_num + tp_num)) + "\n"
            f.write(save_str)

        total_video = u"全部视频"
        total_video = total_video.encode("gb2312")
        save_str = str(g_model_name) + "," + str(conf_name.split(" ")[-1]) + "," + str(conf_name.split(" ")[0]) + "," + \
                       str(g_gt_obj_size_list[inx].encode("gb2312")) + "," + str(total_video) + ","
        save_str += str(total_gt_num) + "," + str(total_tp_num + total_fp_num) + "," + \
                str(total_tp_num) + "," + str(total_fp_num) + "," + \
                str(float(total_tp_num) / max(0.00001, total_gt_num)) + "," + \
                str(float(total_tp_num) / max(0.00001, total_fp_num + total_tp_num)) + "\n"
        f.write(save_str)


def pre_recall_process(match_result_dict, no_match_result_dict, label_result_dict, conf_name, size_inx):

    video_pre_recall_list = video_pre_recall_process(match_result_dict, no_match_result_dict,\
                                                     label_result_dict, float(conf_name.split(" ")[0]), size_inx)
    save_video_pre_recall_to_csv(video_pre_recall_list, conf_name, size_inx)

    if SCENE_PRERECALL:
        scene_pre_recall_dict = scene_pre_recall_process(video_pre_recall_list)
        save_scene_pre_recall_to_csv(scene_pre_recall_dict, conf_name, size_inx)


def Debug_Info(debug_info):
    print debug_info


def get_fppi_recall_list(match_result_dict, no_match_result_dict, det_pic_num, all_gt_num):

    fppi_recall_list = []
    fppi_recall_fppi_list = np.arange(0, 1, 0.01)
    fppi_recall_recall_list = []

    fppi_recall_conf_list = get_fppi_conf_list_by_fppi(no_match_result_dict, det_pic_num, fppi_recall_fppi_list)
    for inx in xrange(len(fppi_recall_conf_list)):
        conf = float(fppi_recall_conf_list[inx])
        tp_num = 0
        for img_inx in match_result_dict.keys():
            for obj in (match_result_dict[img_inx]):
                if float(obj.split(" ")[8]) >= conf:
                    tp_num += 1
        recall = float(tp_num) / all_gt_num
        fppi_recall_recall_list.append(recall)

    # 截断后面的recall
    max_value = max(fppi_recall_recall_list)
    max_index = fppi_recall_recall_list.index(max_value)
    fppi_recall_fppi_list = fppi_recall_fppi_list[:max_index + 1]
    fppi_recall_recall_list = fppi_recall_recall_list[:max_index + 1]

    fppi_recall_list.append(fppi_recall_fppi_list)
    fppi_recall_list.append(fppi_recall_recall_list)
    return fppi_recall_list


def get_all_gt_num_per_type(label_dict):
    gt_num = 0
    for key in label_dict.keys():
        gt_num += len(label_dict[key])
    return gt_num


def get_iou_recall_list(match_result_dict,all_gt_num):
    iou_recall_list = []
    iou_recall_iou_list = np.arange(0.0, 1, 0.01)
    iou_recall_recall_list = []

    iou_result_list = []
    for img_name in match_result_dict.keys():
        obj_list = match_result_dict[img_name]
        for obj in obj_list:
            iou_result_list.append(float(obj.split(" ")[9]))  # overlap
    sorted(iou_result_list, reverse=True)
    for iou_value in iou_recall_iou_list:
        num = 0
        for inx in xrange(len(iou_result_list)):
            if iou_result_list[inx] > iou_value:
                num += 1
        iou_recall_recall_list.append(float(num) / all_gt_num)
    iou_recall_list.append(iou_recall_iou_list)
    iou_recall_list.append(iou_recall_recall_list)
    return iou_recall_list


def get_fppi_conf_list(no_match_result_dict, det_pic_num):
    fppi_conf_list = []
    fppi_conf_fppi_list = np.arange(0, 1, 0.01)
    fppi_conf_conf_list = get_fppi_conf_list_by_fppi(no_match_result_dict, det_pic_num, fppi_conf_fppi_list)

    # 截断
    min_conf = min(fppi_conf_conf_list)
    min_index = fppi_conf_conf_list.index(min_conf)
    fppi_conf_fppi_list = fppi_conf_fppi_list[:min_index + 1]
    fppi_conf_conf_list = fppi_conf_conf_list[:min_index + 1]

    fppi_conf_list.append(fppi_conf_fppi_list)
    fppi_conf_list.append(fppi_conf_conf_list)
    return fppi_conf_list


def get_conf_recall_list(match_result_dict, all_gt_num):
    conf_recall_list = []
    conf_recall_conf_list = np.arange(0.0, 1, 0.01)
    conf_recall_recall_list = []

    conf_result_list = []
    for img_name in match_result_dict.keys():
        obj_list = match_result_dict[img_name]
        for obj in obj_list:
            conf_result_list.append(float(obj.split(" ")[8]))  # conf
    sorted(conf_result_list, reverse=True)
    for conf_value in conf_recall_conf_list:
        num = 0
        for inx in xrange(len(conf_result_list)):
            if conf_result_list[inx] > conf_value:
                num += 1
        conf_recall_recall_list.append(float(num) / all_gt_num)
    conf_recall_list.append(conf_recall_conf_list)
    conf_recall_list.append(conf_recall_recall_list)
    return conf_recall_list


def get_performance_curve_result(match_result_dict, no_match_result_dict, det_pic_num, label_dict):

    result_value_list = []
    fppi_recall_dict = {}
    iou_recall_dict = {}
    fppi_conf_dict = {}
    conf_recall_dict = {}

    all_gt_num = get_all_gt_num_per_type(label_dict)

    # fppi 曲线(fppi--recall)
    fppi_recall_list = get_fppi_recall_list(match_result_dict, no_match_result_dict, det_pic_num, all_gt_num)
    fppi_recall_dict[g_curve_turple[0]] = fppi_recall_list

    # iou 曲线(iou--recall)
    iou_recall_list = get_iou_recall_list(match_result_dict, all_gt_num)
    iou_recall_dict[g_curve_turple[1]] = iou_recall_list

    # score 曲线(fppi--conf)
    fppi_conf_list = get_fppi_conf_list(no_match_result_dict, det_pic_num)
    fppi_conf_dict[g_curve_turple[2]] = fppi_conf_list

    # conf-recall曲线
    conf_recall_list = get_conf_recall_list(match_result_dict, all_gt_num)
    conf_recall_dict[g_curve_turple[3]] = conf_recall_list

    result_value_list.append(fppi_recall_dict)
    result_value_list.append(iou_recall_dict)
    result_value_list.append(fppi_conf_dict)
    result_value_list.append(conf_recall_dict)

    return result_value_list


def save_curve_per_type(inx, performance_result_dict, obj_type, model_list):
    plt.figure()
    plt.grid()
    plt.xticks(np.arange(0, 1.1, 0.05))
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    for key in performance_result_dict.keys():
        print key

    for model_inx in xrange(len(model_list)):
        model_name = model_list[model_inx].strip()
        print model_name
        #print performance_result_dict
        # model_name = performance_result_dict.keys()[model_inx]
        result_list = performance_result_dict[model_name]
        per_type_result = result_list[inx]
        type_name = per_type_result.keys()[0]
        plt.plot(per_type_result[type_name][0], per_type_result[type_name][1], label=model_name,
                 color=g_color_turple[model_inx % (len(g_color_turple))], linewidth=2)
        plt.title(obj_type + '--' + type_name.split('_')[0])
        plt.xlabel(type_name.split('_')[1])
        plt.ylabel(type_name.split('_')[2])
        plt.legend()
    plt.savefig(obj_type + '-' + type_name + '.jpg')
    plt.show()


def save_performance_result(performance_result_dict, obj_type, modle_list):
    for inx in xrange(len(g_curve_turple)):
        print obj_type, g_curve_turple[inx]
        save_curve_per_type(inx, performance_result_dict, obj_type, modle_list)


def get_curve_performance(obj_type, model_list, filter_obj_size):

    if 0 == len(modle_list):
    # if 1:
        detect_result_file = "./model_result\\"
        detect_txt_list = glob.glob(os.path.join(detect_result_file, "*/*.txt"))
    else:
        detect_txt_file = ["./model_result\\" + modle_list[i] for i in xrange(len(modle_list))]
        detect_txt_list = []
        for i in xrange(len(detect_txt_file)):
            # path = detect_txt_file[i] + "/"
            txt_list = glob.glob(os.path.join(detect_txt_file[i] + "/", "*.txt"))
            detect_txt_list += txt_list

        # detect_txt_list += [glob.glob(os.path.join(detect_txt_file[i] + "/", "*.txt")) for i in xrange(len(detect_txt_file))]

    label_result_file = "./bin/label_result/"
    label_txt_list = glob.glob(os.path.join(label_result_file, "*.txt"))

    global dst_file
    dst_file = dst_file[:-1] + "-" + obj_type + '/'
    if os.path.exists(dst_file[:-1]):
        shutil.rmtree(dst_file[:-1])
    if not os.path.exists(dst_file[:-1]):
        os.makedirs(dst_file[:-1])

    # label_dict_all_type_origion = get_label_dict_all_type(label_txt_list)
    Debug_Info("init label_dict finish")

    performance_result_dict = {}
    for txt in detect_txt_list:
        if not os.path.exists(txt):
            print "no this file", txt
            os.system("pause")

        if not txt.split("\\")[-1].startswith(obj_type):
            continue
        model_name = txt.strip().split("\\")[-2]
        Debug_Info(model_name)

        label_dict_all_type = get_label_dict_all_type(label_txt_list)
        #print label_dict_all_type

        # 检测结果字典
        detect_result_dict = get_detect_result(txt)
        Debug_Info("init detect_result dict")

        #print detect_result_dict
        #print obj_type
        get_match_label_detect_dict(label_dict_all_type, detect_result_dict, obj_type)
        det_pic_num = len(detect_result_dict)
        #print detect_result_dict
        # 获得与label匹配上的目标信息
        # match_result_dict:for per img: label_rect + detect_rect + conf + overlap
        # no_match_result_dict: for per img: detect_rect + conf + no_match_type
        Debug_Info("match procseeing...")
        match_result_dict, no_match_result_dict = get_match_result(label_dict_all_type, detect_result_dict, obj_type)
        Debug_Info("match finished")

        label_dict_all_type, match_result_dict = filter_label_detect_result_by_size(label_dict_all_type, match_result_dict, obj_type, filter_obj_size)

        #print match_result_dict
        per_performance_result = get_performance_curve_result(match_result_dict, no_match_result_dict, det_pic_num,
                                                              label_dict_all_type[obj_type])
        performance_result_dict[model_name] = per_performance_result
        Debug_Info("performance result dict finished")
    save_performance_result(performance_result_dict, obj_type, model_list)


def filter_label_detect_result_by_size(label_dict_all_type, match_result_dict, obj_type, filter_size):

    # filter label dict
    label_dict_temp = label_dict_all_type[obj_type]
    labels_keys = label_dict_temp.keys()
    for key in labels_keys:
        delete_list = []
        for cnt in xrange(len(label_dict_temp[key])):
            info = label_dict_temp[key][cnt]
            rect = info.split(" ")
            if abs(int(rect[2]) - int(rect[0])) <= filter_size:
                delete_list.append(info)
        for info in delete_list:
            # print info
            label_dict_all_type[obj_type][key].remove(info)

    # filter match result dict
    for key in match_result_dict.keys():
        delete_list = []
        for cnt in xrange(len(match_result_dict[key])):
            info = match_result_dict[key][cnt]
            rect = info.split(" ")
            if abs(int(rect[2]) - int(rect[0])) <= filter_size:
                delete_list.append(info)
        for info in delete_list:
            # print info
            match_result_dict[key].remove(info)
    return label_dict_all_type, match_result_dict


def get_all_size_performance_process(obj_type, modle_list, filter_obj_size):
    if 0 == len(modle_list):
        detect_result_file = "./model_result\\"
        detect_txt_list = glob.glob(os.path.join(detect_result_file, "*/*.txt"))
    else:
        detect_txt_file = ["./model_result\\" + modle_list[i] for i in xrange(len(modle_list))]
        detect_txt_list = []
        for i in xrange(len(detect_txt_file)):
            txt_list = glob.glob(os.path.join(detect_txt_file[i] + "/", "*.txt"))
            detect_txt_list += txt_list

    label_result_file = "./bin/label_result/"
    label_txt_list = glob.glob(os.path.join(label_result_file, "*.txt"))

    # 获得带求取目标的fppi
    global g_fppi_category, g_fppi_category_person, g_fppi_category_vehicle
    g_fppi_category = g_fppi_category_person if obj_type == "person" else g_fppi_category_vehicle

    global dst_file
    dst_file = dst_file[:-1] + "-" + obj_type + '/'
    if os.path.exists(dst_file[:-1]):
        shutil.rmtree(dst_file[:-1])
    if not os.path.exists(dst_file[:-1]):
        os.makedirs(dst_file[:-1])

    init_csv_file(dst_file)
    init_video_type_dict()
    # label_dict_all_type_origion = get_label_dict_all_type(label_txt_list)
    global g_gt_obj_size_list
    g_gt_obj_size_list = get_gt_obj_size_list()

    for txt in detect_txt_list:
        if not os.path.exists(txt):
            print "no this file", txt
            continue
        if not txt.split("\\")[-1].startswith(obj_type):
            continue
        global g_model_name
        g_model_name = obj_type + "-" + txt.strip().split("\\")[-2].encode("gb2312")
        print g_model_name

        label_dict_all_type = get_label_dict_all_type(label_txt_list)

        # 检测结果字典
        detect_result_dict = get_detect_result(txt)
        # print det_pic_num, len(detect_result_dict)

        get_match_label_detect_dict(label_dict_all_type, detect_result_dict, obj_type)
        det_pic_num = len(detect_result_dict)

        # 获得与label匹配上的目标信息
        match_result_dict, no_match_result_dict = get_match_result(label_dict_all_type, detect_result_dict, obj_type)
        # print len(match_result_dict), len(no_match_result_dict)

        label_dict_all_type, match_result_dict = filter_label_detect_result_by_size(label_dict_all_type, match_result_dict, obj_type, filter_obj_size)

        # 获得fppi对应的conf
        fppi_conf_list = get_fppi_conf_list_by_fppi(no_match_result_dict, det_pic_num, g_fppi_category)
        test_conf_list = g_conf_list + fppi_conf_list

        for conf_inx in xrange(len(test_conf_list)):
            if conf_inx >= len(g_conf_list):
                conf_name = str(test_conf_list[conf_inx]) + " fppi=" + \
                            str(g_fppi_category[conf_inx - len(g_conf_list)])
            else:
                conf_name = str(test_conf_list[conf_inx]) + " " + str(test_conf_list[conf_inx])
            for size_inx in xrange(len(g_gt_obj_size_list)):
                pre_recall_process(match_result_dict, no_match_result_dict, label_dict_all_type[obj_type], conf_name,
                                   size_inx)


def get_model_test_list(model_test_list):
    model_list = []
    for model_name in open(model_test_list, "r").readlines():
        model_name = model_name.strip()
        model_list.append(model_name)
    return model_list

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "Parameter Error!"
        sys.exit()
    obj_type = sys.argv[1]
    func = sys.argv[2]
    #filter_obj_size = int(sys.argv[3])

    filter_obj_size = 0
    # obj_type = 'person'
    # func = '2'

    global OVERLAP
    if obj_type == 'person':
        OVERLAP = 0.4
    print OVERLAP

    model_test_list = "model_test_list.txt"
    if os.path.exists(model_test_list):
        modle_list = get_model_test_list(model_test_list)
    else:
        modle_list = []

    if func == '1':
        get_all_size_performance_process(obj_type, modle_list, filter_obj_size)
    if func == '2':
        get_curve_performance(obj_type, modle_list, filter_obj_size)



