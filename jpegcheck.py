# -*- coding: utf-8 -*-
import os
#import cv2
import shutil
import sys,md5
import numpy as np
#from matplotlib import pyplot as plt
#from pdb import set_trace
import click

reload(sys)
sys.setdefaultencoding('utf-8')



def jpeg_check(path):
    s=[]
    files= os.listdir(path) #得到文件夹下的所有文件名称
    for file in files: #遍历文件夹
         file_path = os.path.join(path,file)
         if not os.path.isdir(file_path): #判断是否是文件夹，不是文件夹才打开

            # use jpeginfo to check jpg image first.
            jpgcheck = 'jpeginfo' +' -c ' + file_path + ' -d' +'\n'
            os.system(jpgcheck)

            if os.path.isfile(file_path):
                #change file houzhui
                if os.path.splitext(file_path)[1] != ".jpg":
                    newname = os.path.splitext(file_path)[0]+".jpg" #要改的新后缀
                    os.rename(file_path,newname)

            else:
               print "%s is deleted\n"%(file_path)
         else:
             jpeg_check(file_path)
    return s


@click.command()
@click.option('--input_dir', default='./food_classify', help='root path of image folder')
#@click.option('--same_data_dir', default='./food_same', help='file search pattern for glob')
#def main(input_dir, same_data_dir):
    #compare_and_move(input_dir,same_data_dir)
def main(input_dir):
    jpeg_check(input_dir)


if __name__ == '__main__':
    main()


