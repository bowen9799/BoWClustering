# -*- coding:UTF-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys,os
import time

import numpy as np
import tensorflow as tf
from glob import glob
from os import path
import logging
import shutil

sys.path.append(".")
#sys.path.append("/home/lhj/PHICOMM/Project/label_model/modelTest/label_model")

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(input_height=299, input_width=299,
        input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"

  # [NEW] make file_name as a placeholder.
  file_name_placeholder = tf.placeholder("string", name="fname")

  file_reader = tf.read_file(file_name_placeholder, input_name)
#  if file_name.endswith(".png"):
#    image_reader = tf.image.decode_png(file_reader, channels = 3,
#                                       name='png_reader')
#  elif file_name.endswith(".gif"):
#    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
#                                                  name='gif_reader'))
#  elif file_name.endswith(".bmp"):
#    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
#  else:
#    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
#                                        name='jpeg_reader')
  image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  #sess = tf.Session()
  #result = sess.run(normalized)
  #return result
  return normalized

def sort_dict(dict_words,index):
    """
    字典排序
    :param dict_words:
    :return:
    """
    keys = dict_words.keys()
    values = dict_words.values()

    list_one = [(key, val) for key, val in zip(keys, values)]
    if index == "value":
      list_sort = sorted(list_one, key=lambda x: x[1], reverse=True)
    else:
      list_sort = sorted(list_one, key=lambda x: x[0], reverse=True)

    return list_sort

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print ("move %s -> %s"%( srcfile,dstfile))

def renameAndMovefile(srcfile,dstfile,prob):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        print(os.path.splitext(srcfile)[0],os.path.splitext(srcfile)[1])

        #change file houzhui
        newname = os.path.splitext(srcfile)[0]+"__"+str("%.2f"%prob)+os.path.splitext(srcfile)[1] #要改的新后缀
        os.rename(srcfile,newname)
        shutil.move(newname,dstfile)          #移动文件
        print ("move %s -> %s"%( newname,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s"%( srcfile,dstfile))

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  label_test = 0
  label_test_ref = 0.99
  label_test_ref_bottom = 0.05
  image_path = "./dataset"
  #path_pattern = "*.[jJ][Pp]*"
  path_pattern = "*.jpg"
  #lable_test_fail_dir = "./dataset/potato_total/label_test_fail_img/"
  #lable_test_check_dir = "./dataset/potato_total/label_test_check_img/"
  #lable_test_pass_dir = "./dataset/potato_total/label_test_pass_img"
  process_mode = "move"


  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  parser.add_argument("--label_test", help="name of label test")
  parser.add_argument("--label_test_ref", help="name of label test reference")
  parser.add_argument("--label_test_ref_bottom", help="name of label test reference")
  parser.add_argument("--image_path", help="image_path to be processed")
  parser.add_argument("--pattern", help="ile search pattern for glob")

  parser.add_argument("--lable_test_fail_dir", help="lable_test_fail_dir")
  parser.add_argument("--lable_test_check_dir", help="lable_test_check_dir")
  parser.add_argument("--lable_test_pass_dir", help="lable_test_pass_dir")
  parser.add_argument("--process_mode", help="copy or move")

  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
#  if args.image:
#    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  if args.label_test:
    label_test = args.label_test
  if args.label_test:
    label_test_ref = args.label_test_ref
  if args.label_test_ref_bottom:
    label_test_ref_bottom = args.label_test_ref_bottom

  if args.image_path:
    image_path = args.image_path
  if args.pattern:
    path_pattern = args.pattern

  if args.lable_test_fail_dir:
    lable_test_fail_dir = args.lable_test_fail_dir
  if args.lable_test_check_dir:
    lable_test_check_dir = args.lable_test_check_dir
  if args.lable_test_pass_dir:
    lable_test_pass_dir = args.lable_test_pass_dir

  if lable_test_pass_dir[len(lable_test_pass_dir)-1] != '/':
    lable_test_pass_dir = lable_test_pass_dir + '/'
  if lable_test_check_dir[len(lable_test_check_dir)-1] != '/':
    lable_test_check_dir = lable_test_check_dir + '/'
  if lable_test_fail_dir[len(lable_test_fail_dir)-1] != '/':
    lable_test_fail_dir = lable_test_fail_dir + '/'

  if args.process_mode:
    process_mode = args.process_mode

  # 获取logger实例，如果参数为空则返回root logger
  logger = logging.getLogger("labelimage")

  # 指定logger输出格式
  formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

  # 文件日志
  #file_handler = logging.FileHandler("test.log")
  now = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
  log_file_name = str("log_") + str(now) + str(".csv")
  file_handler = logging.FileHandler(log_file_name)
  file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式

  # 控制台日志
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.formatter = formatter  # 也可以直接给formatter赋值

  # 为logger添加的日志处理器
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)

  # 指定日志的最低输出级别，默认为WARN级别
  logger.setLevel(logging.DEBUG)

  print('This is debug message')


  current_pwd = os.getcwd()

  #get true file
  true_image_path = path.join(image_path,'true')
  os.chdir(true_image_path)
  true_file = glob(path_pattern)
  #logger.debug("true_file:%s" %(true_file))
  #print(true_file)
  os.chdir(current_pwd)

  #get false file
  false_image_path = path.join(image_path,'false')
  os.chdir(false_image_path)
  false_file = glob(path_pattern)
  #logger.debug("false_file:%s" %(false_file))
  #print(false_file)
  os.chdir(current_pwd)


  all_files = glob(path.join(true_image_path, path_pattern))
  all_files = all_files + glob(path.join(false_image_path, path_pattern))
  #all_files.sort()
  print('Found {} files in {} folder'.format(len(all_files), image_path))
  #logger.debug("all_files:%s" %(all_files))
  #print(all_files)

  graph = load_graph(model_file)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  file_probability = {}
  file_probability_aftersort = {}
  lable_test_pass_num_total = 0
  lable_test_pass_num_true = 0
  lable_test_pass_num_false = 0
  lable_test_fail_num_total = 0
  lable_test_fail_num_true = 0
  lable_test_fail_num_false = 0
  lable_test_check_num_total = 0
  lable_test_check_num_true = 0
  lable_test_check_num_false = 0


  start = time.time()
  with tf.Session(graph=graph) as sess:
    read_tensor_from_image_file_op = read_tensor_from_image_file(
                           input_height=input_height,
                           input_width=input_width,
                           input_mean=input_mean,
                           input_std=input_std)
    for file_name in all_files:

      t = sess.run(read_tensor_from_image_file_op,feed_dict={"fname:0": file_name})
      results = sess.run(output_operation.outputs[0],
                       {input_operation.outputs[0]: t})
      #end=time.time()
      results = np.squeeze(results)
      top_k = results.argsort()[-5:][::-1]
      labels = load_labels(label_file)
      #logger.debug('\nEvaluation image:'+str(file_name))
      #print('\nEvaluation image:'+str(file_name))
      #print('Evaluation time (1-image): {:.3f}s\n'.format(end-start))
      label_index = 0
      for i in top_k:
      #logger.debug("label: %s , %.2f ",labels[top_k[i]], results[top_k[i]])
        #print("label: %s , %.2f "%(labels[top_k[i]], results[top_k[i]]))
        if str(labels[top_k[i]]) == label_test:
         label_index = i
       #logger.debug("evaluating label: %s , %d " % (str(labels[top_k[label_index]]), label_index))
          #print("evaluating label: %s , %d " % (str(labels[top_k[label_index]]), label_index))
      #logger.debug(" label_eval:%s  probability:%.2f  ExpectLabel:%s Hthresh:%s Lthresh:%s" % (labels[top_k[label_index]], results[top_k[label_index]], label_test, label_test_ref,label_test_ref_bottom))
      #print(" label_eval:%s  probability:%.2f  ExpectLabel:%s Hthresh:%s Lthresh:%s" % (labels[top_k[label_index]], results[top_k[label_index]], label_test, label_test_ref,label_test_ref_bottom))
      file_probability[file_name] = results[top_k[label_index]]

      if process_mode == "move":
          if (float(results[top_k[label_index]]) >= float(label_test_ref)):
            lable_test_pass_num_total = lable_test_pass_num_total + 1.0
            if file_name.split('/')[-1] in true_file:
              lable_test_pass_num_true = lable_test_pass_num_true + 1.0
              mymovefile(file_name, lable_test_pass_dir + "true/" )
            else:
              lable_test_pass_num_false = lable_test_pass_num_false + 1.0
              mymovefile(file_name, lable_test_pass_dir + "false/" )

          elif (float(results[top_k[label_index]]) <= float(label_test_ref_bottom)):
            lable_test_fail_num_total = lable_test_fail_num_total + 1.0
            if file_name.split('/')[-1] in true_file:
              lable_test_fail_num_true = lable_test_fail_num_true + 1.0
              mymovefile(file_name, lable_test_fail_dir + "true/")
            else:
              lable_test_fail_num_false = lable_test_fail_num_false + 1.0
              mymovefile(file_name, lable_test_fail_dir + "false/" )
            #print(file_name.split('/')[-1])
          else:
            lable_test_check_num_total = lable_test_check_num_total + 1.0
            if file_name.split('/')[-1] in true_file:
              lable_test_check_num_true = lable_test_check_num_true + 1.0
              mymovefile(file_name, lable_test_check_dir + "true/" )
            else:
              lable_test_check_num_false = lable_test_check_num_false + 1.0
              mymovefile(file_name, lable_test_check_dir + "false/" )

      else:
        image_name = file_name.split("/")[-1]
        if (float(results[top_k[label_index]]) >= float(label_test_ref)):
            lable_test_pass_num_total = lable_test_pass_num_total + 1.0
            if file_name.split('/')[-1] in true_file:
                lable_test_pass_num_true = lable_test_pass_num_true + 1.0
                mycopyfile(file_name, lable_test_pass_dir + "true/" + "/" + image_name)
            else:
                lable_test_pass_num_false = lable_test_pass_num_false + 1.0
                mycopyfile(file_name, lable_test_pass_dir + "false/" + "/" + image_name)

        elif (float(results[top_k[label_index]]) <= float(label_test_ref_bottom)):
            lable_test_fail_num_total = lable_test_fail_num_total + 1.0
            if file_name.split('/')[-1] in true_file:
                lable_test_fail_num_true = lable_test_fail_num_true + 1.0
                mycopyfile(file_name, lable_test_fail_dir + "true/"+ "/" + image_name)
            else:
                lable_test_fail_num_false = lable_test_fail_num_false + 1.0
                mycopyfile(file_name, lable_test_fail_dir + "false/" + "/" + image_name)
                #print(file_name.split('/')[-1])
        else:
            lable_test_check_num_total = lable_test_check_num_total + 1.0
            if file_name.split('/')[-1] in true_file:
                lable_test_check_num_true = lable_test_check_num_true + 1.0
                mycopyfile(file_name, lable_test_check_dir + "true/" + "/" + image_name)
            else:
                lable_test_check_num_false = lable_test_check_num_false + 1.0
                mycopyfile(file_name, lable_test_check_dir + "false/" + "/" + image_name)

      #print(file_name.split('/')[-1])

  lable_test_pass_num_true_rate = lable_test_pass_num_true / lable_test_pass_num_total
  lable_test_fail_num_true_rate = lable_test_fail_num_true / lable_test_fail_num_total
  lable_test_check_num_true_rate = lable_test_check_num_true / lable_test_check_num_total
  #print(file_probability)
  file_probability_aftersort = sort_dict(file_probability,"key")
  #print(file_probability_aftersort)
  logger.debug("file name,probability,label")
  for key,value in file_probability_aftersort:
    if key.split('/')[-1] in true_file:
      logger.debug("%s,%s,1" %(key,value))
    else:
      logger.debug("%s,%s,0" %(key,value))

  end=time.time()
  logger.debug('Evaluation time (1-image): {:.3f}s\n'.format(end-start))

  logger.debug("---------------------------")
  logger.debug("pass num:%d, true = %d (%.3f), false = %d (%.3f)" \
      %(lable_test_pass_num_total,lable_test_pass_num_true,lable_test_pass_num_true_rate,\
        lable_test_pass_num_false,(1-lable_test_pass_num_true_rate)))
  logger.debug("check num:%d, true = %d (%.3f), false = %d (%.3f)" \
      %(lable_test_check_num_total,lable_test_check_num_true,lable_test_check_num_true_rate,\
        lable_test_check_num_false,(1-lable_test_check_num_true_rate)))
  logger.debug("fail num:%d, true = %d (%.3f), false = %d (%.3f)" \
      %(lable_test_fail_num_total,lable_test_fail_num_true,lable_test_fail_num_true_rate,\
        lable_test_fail_num_false,(1-lable_test_fail_num_true_rate)))

  # 移除一些日志处理器
  logger.removeHandler(file_handler)









