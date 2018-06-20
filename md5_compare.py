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


def classify_gray_hist(hist1,hist2):
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
        else:
            degree = degree + 1
    degree = degree/len(hist1)
    return degree

def scan_folder(path,s):
    files= os.listdir(path) #得到文件夹下的所有文件名称
    for file in files: #遍历文件夹
         file_path = os.path.join(path,file)
         if not os.path.isdir(file_path): #判断是否是文件夹，不是文件夹才打开

            # use jpeginfo to check jpg image first.
            jpgcheck = 'jpeginfo' +' -c ' + file_path + ' -d' +'\n'
            #print "jpgpath %s"%(jpgcheck)
            #sys.stdout.write('\n')
            #sys.stdout.flush()
            os.system(jpgcheck)
            if os.path.isfile(file_path):
               f1 = open(file_path,'r')
               md5_value = md5.new(f1.read()).digest()
               s.append([path+"/"+file,md5_value])
            else:
               print "%s is deleted\n"%(file_path)
         else:
             scan_folder(file_path,s)
    return s

def scan_folder_and_remove_same(path):
    s=[]
    files= os.listdir(path) #得到文件夹下的所有文件名称
    for file in files: #遍历文件夹
         file_path = os.path.join(path,file)
         if not os.path.isdir(file_path): #判断是否是文件夹，不是文件夹才打开

            if False:
                # use jpeginfo to check jpg image first.
                jpgcheck = 'jpeginfo' +' -c ' + file_path + ' -d' +'\n'
                os.system(jpgcheck)

                if os.path.isfile(file_path):

                    f1 = open(file_path,'r')
                    md5_value = md5.new(f1.read()).digest()
                    if md5_value in s :
                        os.remove(file_path)
                    else:
                        s.append(md5_value)

                        #change file houzhui
                        if os.path.splitext(file_path)[1] != ".jpg":
                            newname = os.path.splitext(file_path)[0]+".jpg" #要改的新后缀
                            os.rename(file_path,newname)
                else:
                   print "%s is deleted\n"%(file_path)
            else:
                f1 = open(file_path,'r')
                md5_value = md5.new(f1.read()).digest()
                if md5_value in s :
                    os.remove(file_path)
                else:
                    s.append(md5_value)
         else:
             scan_folder_and_remove_same(file_path)
    return s


def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print "move %s -> %s"%( srcfile,dstfile)

def compare_and_move(path, compared_path):
    file_list = []
    scan_folder(path,file_list)
    s_images = []
    print(path)

    for i in range(len(file_list)-1):
        #print(i)
        for j in range(i+1,len(file_list)):
            if file_list[i][1]==file_list[j][1]:
                s_images.append(file_list[j][0])
                print(file_list[i][0],file_list[j][0])
    s_images_path = list(set(s_images))
    for s_image_path in s_images:
        s_image = s_image_path.split("/")[-1]
        mymovefile(s_image_path, compared_path + "/" + s_image)

def compare_and_delete(path):
    scan_folder_and_remove_same(path)

@click.command()
@click.option('--input_dir', default='./food_classify', help='root path of image folder')
#@click.option('--same_data_dir', default='./food_same', help='file search pattern for glob')
#def main(input_dir, same_data_dir):
    #compare_and_move(input_dir,same_data_dir)
def main(input_dir):
    compare_and_delete(input_dir)


if __name__ == '__main__':
    main()


