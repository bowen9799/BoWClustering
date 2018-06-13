# -*- coding:utf-8 -*-


"""
根据典型图片, 对目标文件夹中的文件进行相似度排序

python similor_sort_py2.py --dir src_dir


src_dir 结构
|--- src_dir
├    └──source (用户图片)
├    └──classic(典型图片)
├    └──sorted (排序后的图片)
├    └──data (restore data, 训练生成)
├    └──model (restore model, 训练生成)
"""

import turicreate as tc
import argparse
import os
import time
from turicreate import SFrame
import pandas as pd
import numpy as np
tc.config.set_num_gpus(0)

def similor_sort(sourceData,classicData, num):
    """
    :param sourceData: dataframe include
    :param classicData: classic picture
    :param num: how many picture to pick out
    :return:
    """
    start_time = time.time()

    ref_data = SFrame()
    for index, row in sourceData.iterrows():
        #print row
        path = row['path']
        img = tc.Image(path)
        ref_data = ref_data.append(SFrame({'path':[path],'image':[img]}))
    ref_data = ref_data.add_row_number()

    # print ref_data

    query_data = SFrame()
    for index, row in classicData.iterrows():
        path = row['path']
        img = tc.Image(path)
        query_data = query_data.append(SFrame({'path':[path],'image':[img]}))
    query_data = query_data.add_row_number()

    model = tc.image_similarity.create(ref_data, label=None, feature=None, model='resnet-50', verbose=True)
    if num == 0:
        num = ref_data.num_rows()


    similar_images = model.query(query_data, k=num)

    ret_array = np.zeros((query_data.num_rows(), num))
    for image in similar_images:
        ref_label = image['reference_label']
        distance = image['distance']
        query_label = image['query_label']
        ret_array[query_label][ref_label] = distance;

    mean = np.mean(ret_array, axis=0)
    sourceData.insert(2,'distance',(mean))
    #sort = np.argsort(mean)
    # print sourceData

    elapsed_time = time.time() - start_time
    print ("Time elapsed = %d"%(elapsed_time))
    return sourceData

if __name__ == '__main__':
    dir = "./test/second"
    out_dir = "./out_dir"
    selet_num = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",    help="source to test")
    parser.add_argument("--selet_num",    help="src dir to copy")
    args = parser.parse_args()

    if args.dir:
        dir = args.dir
    if args.selet_num:
        selet_num = int(args.selet_num)

    source_path = os.path.join(dir, "source")
    filelist = os.listdir(source_path)
    df=pd.DataFrame(columns=['name','path'])
    for file in filelist:
        path=os.path.join(source_path,file)
        df=df.append([{'name':file, 'path':path}],ignore_index=True)

    bm_path =  os.path.join(dir,"classic")
    bm_list = os.listdir(source_path)
    bm = pd.DataFrame(columns=['name', 'path'])
    for file in bm_list:
        path=os.path.join(source_path,file)
        bm=df.append([{'name':file, 'path':path}],ignore_index=True)

    similor_sort(df, bm, selet_num)
