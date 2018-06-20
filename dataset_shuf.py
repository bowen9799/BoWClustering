# -*- coding: utf-8 -*-
import os
#import cv2
import shutil
import numpy as np
import click
from glob import glob
from os import path
import sys
import random
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)               #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print "move %s -> %s"%( srcfile,dstfile)

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)               #创建路径
        #print "here %s -> %s"%( srcfile,dstfile)
        shutil.copyfile(srcfile,dstfile)
        print "copy %s -> %s"%( srcfile,dstfile)

def shuf(true_dir, false_dir,shuf_dir,shuf_num):
    if os.path.exists(shuf_dir):
        #sys.exit()
        os.removedirs(shuf_dir)   #删除空文件夹

    all_files = glob(path.join(true_dir, "*.[jJ][Pp]*"))
    all_files = all_files + glob(path.join(false_dir, "*.[jJ][Pp]*"))
    #print(all_files)
    shuf_file = random.sample(all_files, shuf_num)
    print(shuf_file)

    for s_image_path in shuf_file:
        s_image = s_image_path.split("/")[-1]
        s_image_label = s_image_path.split("/")[-2]
        #(s_image,s_image_label)
        mycopyfile(s_image_path, shuf_dir + "/" + s_image_label + "/" + s_image)


@click.command()
@click.option('--true_dir', default='./fengwei_after_clean_total', help='root path of image folder')
@click.option('--false_dir', default='./fengwei_true', help='file search pattern for glob')
@click.option('--shuf_dir', default='./fengwei_true', help='file search pattern for glob')
@click.option('--shuf_num', default=5, help='file search pattern for glob')
def main(true_dir, false_dir,shuf_dir,shuf_num):
    shuf(true_dir,false_dir,shuf_dir,shuf_num)


if __name__ == '__main__':
    main()


