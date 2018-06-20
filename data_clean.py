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
import label_image as image_predict
import retrain as clean_model_create
import yaml
import pandas as pd
import sample_select
sys.path.append(".")

FLAGS = None

def logger_setting():
  # 获取logger实例，如果参数为空则返回root logger
  logger = logging.getLogger("labelimage")

  # 指定logger输出格式
  formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

  # 文件日志
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
  logger.debug('This is debug message')
  return logger, file_handler

def load_config(config_file='config.yaml'):
    ''' 读取配置文件，并返回一个python dict 对象

    :param config_file: 配置文件路径
    :return: python dict 对象
  '''
    with open(config_file, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as e:
            print(e)
            return None
    return config

def searchDataFromDataFrameWithKeyAndValue(df,key,keyValue):
  df2 = df[df.get(key).isin([keyValue])]
  return df2
  pass

def searchDataFromDataFramWithKeyAndNoValue(df,key,keyValue):
    df3 = df[df.get(key) != keyValue]
    return df3
    pass

def main(_):


  if os.path.exists(FLAGS.config) and os.path.isfile(FLAGS.config):
      # load config file
      config = load_config(config_file=FLAGS.config)
      #print(config)
      #coase argment
      coase_arg={}
      coase_arg["model_file"] = config.get('coarse_graph', "tf_files/retrained_graph.pb")
      coase_arg["input_layer"] = config.get('coarse_input_layer', "input")
      coase_arg["output_layer"] = config.get('coarse_output_layer', "final_result")
      coase_arg["input_height"] = config.get('coarse_input_height', 224)
      coase_arg["input_width"] = config.get('coarse_input_width', 224)
      coase_arg["label_test"] = config.get('coarse_label_test', "truefood")
      coase_arg["label_file"] = config.get('coarse_labels', "tf_files/retrained_labels.txt")
      coase_arg["label_test_ref_bottom"] = config.get('coarse_label_test_ref_bottom', 0.05)
      coase_arg["input_mean"] = config.get('coarse_input_mean', 128)
      coase_arg["input_std"] = config.get('coarse_input_std', 128)
      coase_arg["image_path"] = config.get('image_path', "./dataset/凤尾虾_去非菜")
      coase_arg["pattern"] = config.get('pattern', "*/*.[jJPp][PpNn]*")

      #retrain argment
      retrain_arg={}
      retrain_arg["image_dir"] = config.get('image_dir', "train/train_first")
      retrain_arg["output_graph"] = config.get('output_graph', "tf_files/dish_first_pb_inception_v3/retrained_graph.pb")
      retrain_arg["intermediate_output_graphs_dir"] = config.get('intermediate_output_graphs_dir', "/tmp/intermediate_graph/")
      retrain_arg["intermediate_store_frequency"] = config.get('intermediate_store_frequency', 0)
      retrain_arg["output_labels"] = config.get('output_labels', "tf_files/dish_first_pb_inception_v3/retrained_labels.txt")
      retrain_arg["summaries_dir"] = config.get('summaries_dir', "tf_files/training_summaries/inception_v3")
      retrain_arg["how_many_training_steps"] = config.get('how_many_training_steps', 500)
      retrain_arg["learning_rate"] = config.get('learning_rate', 0.01)
      retrain_arg["testing_percentage"] = config.get('testing_percentage', 10)
      retrain_arg["validation_percentage"] = config.get('validation_percentage', 10)
      retrain_arg["eval_step_interval"] = config.get('eval_step_interval', 10)
      retrain_arg["train_batch_size"] = config.get('train_batch_size', 100)
      retrain_arg["test_batch_size"] = config.get('test_batch_size', -1)
      retrain_arg["validation_batch_size"] = config.get('validation_batch_size', 100)
      retrain_arg["print_misclassified_test_images"] = config.get('print_misclassified_test_images', False)
      retrain_arg["model_dir"] = config.get('model_dir', "tf_files/models/")
      retrain_arg["bottleneck_dir"] = config.get('bottleneck_dir', "tf_files/bottlenecks")
      retrain_arg["final_tensor_name"] = config.get('final_tensor_name', "final_result")
      retrain_arg["flip_left_right"] = config.get('flip_left_right', False)
      retrain_arg["random_crop"] = config.get('random_crop', 0)
      retrain_arg["random_scale"] = config.get('random_scale', 0)
      retrain_arg["random_brightness"] = config.get('random_brightness', 0)
      retrain_arg["architecture"] = config.get('fine_grained_architecture', "inception_v3")

      #fine argment
      fine_arg={}
      fine_arg["model_file"] = config.get('fine_graph', "tf_files/retrained_graph.pb")
      fine_arg["input_layer"] = config.get('fine_input_layer', "input")
      fine_arg["output_layer"] = config.get('fine_output_layer', "final_result")
      fine_arg["input_height"] = config.get('fine_input_height', 224)
      fine_arg["input_width"] = config.get('fine_input_width', 224)
      fine_arg["label_test"] = config.get('fine_label_test', "truefood")
      fine_arg["label_file"] = config.get('fine_labels', "tf_files/retrained_labels.txt")
      fine_arg["label_test_ref_bottom"] = config.get('fine_label_test_ref_bottom', 0.05)
      fine_arg["input_mean"] = config.get('fine_input_mean', 128)
      fine_arg["input_std"] = config.get('fine_input_std', 128)
      fine_arg["image_path"] = config.get('image_path', "./dataset/凤尾虾_去非菜")
      fine_arg["pattern"] = config.get('pattern', "*/*.[jJPp][PpNn]*")

      #nearest argment
      nearest_arg = {}
      nearest_arg["nearest_positive_output_dir"] = config.get('nearest_positive_output_dir', "train/train_first/truefood")
      nearest_arg["nearest_negative_output_dir"] = config.get('nearest_negative_output_dir', "train/train_first/falsefood")
      nearest_arg["classic_dir"] = config.get('classic_dir', "classic")
      nearest_arg["classic_pattern"] = config.get('classic_pattern', "*.[jJPp][PpNn]*")
      nearest_arg["nearest_sample_num"] = config.get('nearest_sample_num', 100)
  else:
      exit()

  #log setting
  logger, file_handler= logger_setting()

  all_files = glob(path.join(coase_arg["image_path"], coase_arg["pattern"]))
  all_files_df = pd.DataFrame(columns=['name', 'path','probability'])
  for file_path in all_files:
      file_name = file_path.split("/")[-1]
      all_files_df=all_files_df.append([{'name':file_name, 'path':file_path}],ignore_index=True)
  #print(all_files_df)
  print("all_files_df,num is ",len(all_files_df))
  #all_files.sort()
  #print('Found {} files in {} folder'.format(len(all_files), FLAGS_clean.image_path))
  #print('Found {} files in folder'.format(len(all_files)))


  #coarse model predict
  all_files_df = image_predict.imagepredict("coarse", all_files_df, coase_arg)
  coarse_delete_files_df = searchDataFromDataFrameWithKeyAndValue(all_files_df,'probability',-1)
  coarse_save_files_df = searchDataFromDataFramWithKeyAndNoValue(all_files_df,'probability',-1)
  print("coarse_delete_files_df,num is %d" % (len(coarse_delete_files_df)))
  print("coarse_save_files_df,num is %d" % (len(coarse_save_files_df)))
  #print(coarse_save_files_df)

  classic_files = glob(path.join(nearest_arg["classic_dir"], nearest_arg["classic_pattern"]))
  classic_files_df = pd.DataFrame(columns=['name', 'path','probability'])
  for file_path in classic_files:
      file_name = file_path.split("/")[-1]
      classic_files_df=classic_files_df.append([{'name':file_name, 'path':file_path}],ignore_index=True)
  print("classic_files_df,num is " ,len(classic_files_df))
  #print(classic_files_df)
  sample_select.integrate(coarse_save_files_df,classic_files_df,nearest_arg["nearest_sample_num"], nearest_arg["nearest_positive_output_dir"],  nearest_arg["nearest_negative_output_dir"])

  #fine model create
  clean_model_create.model_retrain("fine",retrain_arg)

  #fine model predict
  coarse_save_files_df = image_predict.imagepredict("fine", coarse_save_files_df, fine_arg)
  print("coarse_save_files_df",coarse_save_files_df)
#  for key,value in file_prob_fine:
#    logger.debug("file name:%s,probability:%s" %(key,value))



  #print.debug('Evaluation time (1-image): {:.3f}s\n'.format(end-start))
  # 移除一些日志处理器
  logger.removeHandler(file_handler)

if __name__ == "__main__":

  #label_image arg
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default="config.yaml", help="graph/model to be executed")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)









