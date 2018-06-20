# -*- coding:utf-8 -*-

from PIL import Image
import os
import click
global allDiff
import shutil
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

allDiff = []
postFix = ()
#---------------------------------
#DHASH实现
#---------------------------------
same_dir_index = 0
def __difference(image):
	"""
	*Private method*
	计算image的像素差值
	:param image: PIL.Image
	:return: 差值数组。0、1组成
	"""
	resize_width = 32
	resize_height = 32
	# 1. resize to (9,8)
	smaller_image = image.resize((resize_width, resize_height))
	# 2. 灰度化 Grayscale
	grayscale_image = smaller_image.convert("L")
	# 3. 比较相邻像素
	pixels = list(grayscale_image.getdata())
	difference = []
	for row in range(resize_height):
		row_start_index = row * resize_width
		for col in range(resize_width - 1):
			left_pixel_index = row_start_index + col
			difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
	return difference

def __hamming_distance_with_hash(dhash1, dhash2):
	"""
	*Private method*
	根据dHash值计算hamming distance
	:param dhash1: str
	:param dhash2: str
	:return: 汉明距离(int)
	"""
	difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
	return bin(difference).count("1")

def calculate_hash(image):
	"""
	计算图片的dHash值
	:param image: PIL.Image
	:return: dHash值,string类型
	"""
	difference = __difference(image)
	# 转化为16进制(每个差值为一个bit,每8bit转为一个16进制)
	decimal_value = 0
	hash_string = ""
	for index, value in enumerate(difference):
		if value:  # value为0, 不用计算, 程序优化
			decimal_value += value * (2 ** (index % 8))
		if index % 8 == 7:  # 每8位的结束
			hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # 不足2位以0填充。0xf=>0x0f
			decimal_value = 0
	return hash_string

def hamming_distance(first, second):
	"""
	计算两张图片的汉明距离(基于dHash算法)
	:param first: Image或者dHash值(str)
	:param second: Image或者dHash值(str)
	:return: hamming distance. 值越大,说明两张图片差别越大,反之,则说明越相似
	"""
	# A. dHash值计算汉明距离
	if isinstance(first, str):
		return __hamming_distance_with_hash(first, second)

	# B. image计算汉明距离
	hamming_distance = 0
	image1_difference = __difference(first)
	image2_difference = __difference(second)
	for index, img1_pix in enumerate(image1_difference):
		img2_pix = image2_difference[index]
		if img1_pix != img2_pix:
			hamming_distance += 1
	return hamming_distance

#--------------------------------

#能够处理的图片后缀
def picPostfix():  # 相册后缀的集合
    postFix = set()
    postFix.update(['bmp', 'jpg', 'png', 'tiff', 'gif', 'pcx', 'tga', 'exif',
                    'fpx', 'svg', 'psd', 'cdr', 'pcd', 'dxf', 'ufo', 'eps', 'JPG', 'raw', 'jpeg'])
    return postFix


def copyfile(file,dest):
    print(file,dest)
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.copy(file, dest)

#判断是否有重复
def foundSame(file_name, hash):

	global same_dir_index
	for i in range(len(allDiff)):
		ans = hamming_distance(allDiff[i][1], hash)
		#print ('[%s]\t[%s]\t%s'%(allDiff[i][0], file_name, ans))
		if ans <= 300:  # 判别的汉明距离，自己根据实际情况设置
			print ('same [%s]\t[%s]\t%s'%(allDiff[i][0], file_name, ans))
			#copyfile(allDiff[i][0],"./same/"+ str(ans)+"/"+str(same_dir_index))
			#copyfile(file_name,"./same/"+str(ans)+"/"+str(same_dir_index))
			#same_dir_index = same_dir_index + 1
			os.remove(file_name)
			return True
	return False
#处理重复图片
def processSame(file_name):
	pass
	#print ('%s is duplicate'%(file_name))

#建立字典
def gethash(file_name):
	global allDiff
	if file_name.split('.')[-1] in postFix:  # 判断后缀是不是照片格式
		try:
			im = Image.open(file_name)
			hash = calculate_hash(im)
			#print('%s  %s'%(file_name, hash))
			if not foundSame(file_name, hash):
				allDiff.append((file_name, hash))
			else:
				processSame(file_name)
		except OSError:
			pass
#遍历文件夹
def processDir(file_dir):
	print('root %s'%(file_dir))
	for file in os.listdir(file_dir):
		file_path = os.path.join(file_dir,file)
		#print(file_path)
		if os.path.isdir(file_path):
			#processDir(r'%s/%s'%(file_dir, file))
			processDir(file_path)
		else:
			gethash(file_path)


@click.command()
@click.option('--input_dir', default='.', help='root path of image folder')
def main(input_dir):
	global postFix
	postFix = picPostfix()  #  图片后缀的集合
	processDir(input_dir)


if __name__ == '__main__':
	main()
