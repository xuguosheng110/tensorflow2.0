#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/12 23:43
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site :简单的数据操作
# @File : character_1.py
# @Software: PyCharm
#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import cv2
import numpy as np

print(tf.__version__)
#1.创建tensor
a = tf.constant(range(12))
print(a,len(a))
a1 = tf.reshape(a,(3,2,2)) #3*2*2 = 12
print(a1)
a2 = a1[np.newaxis,:]
print(a2)
a3 = tf.ones((3,4))
print(a3)
a4 = tf.random.normal(shape=(3,4),mean = 0, stddev=1)
print(a4)
a5 = [tf.range(4)]
print(a5)
a6 = tf.tile(a5,[3,1])
print(a6)
a7 = tf.cast(a6,dtype=tf.float32)
print(a7)
a8 = a4 + a7
print(a8)
a9 = tf.transpose(a8)
print(a9)
a10 = tf.concat([a4,a7],axis=0)
print(a10)
a11 = tf.concat([a4,a7],axis=1)
print(a11)
a12 = tf.reduce_sum(a10,axis=0)
print(a12)
a12 = tf.reduce_sum(a10,axis=1)
print(a12)
a12 = tf.reduce_sum(a10,axis=(0,1))
print(a12)

#broadcast 广播机制 不同纬度数据拷贝成维度一致进行计算
x = tf.reshape(tf.constant(tf.range(3)),(1,3))
y = tf.reshape(tf.constant(tf.range(2)),(2,1))
sum = x+y
print(x,y,sum)

#assign 重新赋值
o = tf.Variable([1,2,3])
o[1].assign(10)
print(o)