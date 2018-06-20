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
    dict sort
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

def imagepredict(precess_mode,files_df,predict_arg):

  model_file = predict_arg["model_file"]
  input_layer = predict_arg["input_layer"]
  output_layer = predict_arg["output_layer"]
  input_height = predict_arg["input_height"]
  input_width = predict_arg["input_width"]
  label_test = predict_arg["label_test"]
  label_file = predict_arg["label_file"]
  label_test_ref_bottom = predict_arg["label_test_ref_bottom"]
  input_mean = predict_arg["input_mean"]
  input_std = predict_arg["input_std"]

  graph = load_graph(model_file)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    read_tensor_from_image_file_op = read_tensor_from_image_file(
                           input_height=input_height,
                           input_width=input_width,
                           input_mean=input_mean,
                           input_std=input_std)

    for index, row in files_df.iterrows():
      #start = time.time()

      file_path = row['path']
      t = sess.run(read_tensor_from_image_file_op,feed_dict={"fname:0": file_path})
      #t = sess.run(read_tensor_from_image_file_op,feed_dict={file_name_placeholder: file_path})
      results = sess.run(output_operation.outputs[0],
                       {input_operation.outputs[0]: t})
      #end=time.time()
      results = np.squeeze(results)
      top_k = results.argsort()[-5:][::-1]
      labels = load_labels(label_file)
      #logger.debug('\nEvaluation image:'+str(file_path))
      #print('\nEvaluation image:'+str(file_path))
      #logger.debug('Evaluation time (1-image): {:.3f}s\n'.format(end-start))
      label_index = 0
      for i in top_k:
      #logger.debug("label: %s , %.2f ",labels[top_k[i]], results[top_k[i]])
        #print("label: %s , %.2f "%(labels[top_k[i]], results[top_k[i]]))
        if str(labels[top_k[i]]) == label_test:
         label_index = i

      if (float(results[top_k[label_index]]) <= float(label_test_ref_bottom)):
        #print("this file is %s, probability:%f "%(row['path'],results[top_k[label_index]]))
        row['probability'] = -1
      else:
        row['probability'] = results[top_k[label_index]]

  return files_df




if __name__ == "__main__":
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
  file_handler = logging.FileHandler("test.log")
  file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式

  # 控制台日志
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.formatter = formatter  # 也可以直接给formatter赋值

  # 为logger添加的日志处理器
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)

  # 指定日志的最低输出级别，默认为WARN级别
  logger.setLevel(logging.DEBUG)

  logger.debug('This is debug message')

  all_files = glob(path.join(image_path, path_pattern))
  #all_files.sort()
  print('Found {} files in {} folder'.format(len(all_files), image_path))
  #print(all_files)

  graph = load_graph(model_file)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  file_probability = {}
  file_probability_aftersort = {}
  start = time.time()
  with tf.Session(graph=graph) as sess:
    read_tensor_from_image_file_op = read_tensor_from_image_file(
                           input_height=input_height,
                           input_width=input_width,
                           input_mean=input_mean,
                           input_std=input_std)

    for file_name in all_files:
      #start = time.time()

      t = sess.run(read_tensor_from_image_file_op,feed_dict={"fname:0": file_name})
      #t = sess.run(read_tensor_from_image_file_op,feed_dict={file_name_placeholder: file_name})
      results = sess.run(output_operation.outputs[0],
                       {input_operation.outputs[0]: t})
      #end=time.time()
      results = np.squeeze(results)
      top_k = results.argsort()[-5:][::-1]
      labels = load_labels(label_file)
      #logger.debug('\nEvaluation image:'+str(file_name))
      #print('\nEvaluation image:'+str(file_name))
      #logger.debug('Evaluation time (1-image): {:.3f}s\n'.format(end-start))
      label_index = 0
      for i in top_k:
      #logger.debug("label: %s , %.2f ",labels[top_k[i]], results[top_k[i]])
        #print("label: %s , %.2f "%(labels[top_k[i]], results[top_k[i]]))
        if str(labels[top_k[i]]) == label_test:
         label_index = i
      file_probability[file_name] = results[top_k[label_index]]


  file_probability_aftersort = sort_dict(file_probability,"key")
  #print(file_probability_aftersort)
  for key,value in file_probability_aftersort:
    logger.debug("file name:%s,probability:%s" %(key,value))

  end=time.time()
  logger.debug('Evaluation time (1-image): {:.3f}s\n'.format(end-start))
  # 移除一些日志处理器
  logger.removeHandler(file_handler)









