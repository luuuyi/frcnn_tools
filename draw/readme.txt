分析工具使用说明；

a.生成相应格式的检测结果，格式根据matlab的版本进行确定
  图片名称 类别 数目 conf rect(x1,y1,x2,y2)
  检测结果放在detect_result下，
  运行bin下的filter_result_by_rules.py即可,结果存在filter_detect_result
 
b.性能详细分析
  检测结果放在model_result下，
  运行bin下的analyze_detect_result_process.py并加相应参数即可， 
  生成详细分析结果.csv,位于analyze_result文件夹,需后跟参数：obj_type 1 obj_size
  生成性能曲线结果，                            需后跟参数：obj_type 2 obj_size
  obj_type为person、car等
  测试模型根据model_test_list.txt进行配置
