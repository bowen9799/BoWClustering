#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw
import os
import shutil

def make_regalur_image(img, size = (256, 256)):
	"""
	Resize and RGB-lize the image 
	"""
	return img.resize(size).convert('RGB')


def split_image(img, part_size = (64, 64)):
	"""
	Split one image into 16 smaller ones
	"""
	w, h = img.size
	pw, ph = part_size
	
	assert w % pw == h % ph == 0
	
	return [img.crop((i, j, i+pw, j+ph)).copy() \
				for i in xrange(0, w, pw) \
				for j in xrange(0, h, ph)]


def hist_similar(lh, rh):
	assert len(lh) == len(rh)
	return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) \
	for l, r in zip(lh, rh))/len(lh)


def calc_similar(li, ri):
#	return hist_similar(li.histogram(), ri.histogram())
	return sum(hist_similar(l.histogram(), r.histogram()) \
	for l, r in zip(split_image(li), split_image(ri))) / 16.0
			

def calc_similar_by_path(lf, rf):
	# print lf, rf
	li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
	return calc_similar(li, ri)


def make_doc_data(lf, rf):
	li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
	li.save(lf + '_regalur.png')
	ri.save(rf + '_regalur.png')
	fd = open('stat.csv', 'w')
	fd.write('\n'.join(l + ',' + r for l, r in zip(map(str, li.histogram()), map(str, ri.histogram()))))
#	print >>fd, '\n'
#	fd.write(','.join(map(str, ri.histogram())))
	fd.close()
	li = li.convert('RGB')
	draw = ImageDraw.Draw(li)
	for i in xrange(0, 256, 64):
		draw.line((0, i, 256, i), fill = '#ff0000')
		draw.line((i, 0, i, 256), fill = '#ff0000')
	li.save(lf + '_lines.png')


def compare_all(image_path):
	drive, sbj = os.path.split(image_path)
	# print drive, sbj
	image_list = os.listdir(drive)
	scores = []
	for obj in image_list:
		if not str(obj).startswith("000"):
			continue
		score = calc_similar_by_path(drive + '/' + sbj, drive + '/' + obj)
		scores.append((obj, score))
	# for t in scores:	
	# 	os.rename(drive + '/' + t[0], drive + '/' + str(t[1]) + '_' + t[0])
	return scores


if __name__ == '__main__':
	# path = r'test/TEST%d/%d.JPG'
	# for i in xrange(1, 7):
	# 	print 'test_case_%d: %.3f%%'%(i, \
	# 		calc_similar_by_path('test/TEST%d/%d.JPG'%(i, 1), 'test/TEST%d/%d.JPG'%(i, 2))*100)
	image_path = r'dataset/255/net360__000285.jpg'
	compare_all(image_path)

#	make_doc_data('test/TEST4/1.JPG', 'test/TEST4/2.JPG')