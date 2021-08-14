# -*- coding: utf-8 -*-
# @Project_Name: epidemic_pridiction
# @Time    : 2021/4/19 19:58
# @Author  : miko
# @Site    : 
# @File    : transform_png_ico.py
# @Software: PyCharm

#jpg转ico
import PythonMagick

img = PythonMagick.Image(r'W:\project\pytorch_project\epidemic_pridiction\text\AI.png')

# 这里要设置一下尺寸，不然会报ico尺寸异常错误

img.sample('128x128')

img.write('AI.ico')