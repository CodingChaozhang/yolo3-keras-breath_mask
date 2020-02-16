#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/12 17:24
# @Author  : codingchaozhang
#!/usr/bin/python
# -*- coding:utf-8 -*-
import os


def generate(txt_name, dir, prefix, folder, label):
    files = os.listdir(dir)  # 里面的图片.png
    files.sort()

    listText = open(txt_name, 'a')
    for file in files:
        name = prefix + folder + '/' + file + ' ' + str(int(label)) + '\n'
        listText.write(name)
    listText.close()


test_path = "img/"  # 这里是你的图片的目录
txt_name = 'test_imglist.txt'

if __name__ == '__main__':
    img_list = os.listdir(test_path)
    img_list.sort()
    list_Text = open(txt_name, 'a')
    for img in img_list:
        name = 'img/' + img + '\n'
        list_Text.write(name)
    list_Text.close()

    print("all done")
