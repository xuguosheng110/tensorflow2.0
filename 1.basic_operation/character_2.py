#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/13 15:47
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site :求导
# @File : character_2.py
# @Software: PyCharm
#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np


# x = tf.reshape(tf.Variable(range(4), dtype=tf.float32),(4,1))
x = tf.reshape(tf.Variable(range(4), dtype=tf.float32), (4,1))


with tf.GradientTape() as g:
    g.watch(x)
    y = 2*x*tf.transpose(x)
    dy_dx = g.gradient(y,x)
print(dy_dx)

#多次调用 需要加参数persistent
with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = 2*x*x
    z = 2*y*y
    dy_dx = g.gradient(y,x)
    dz_dx = g.gradient(z,x)
print(dy_dx,dz_dx)

#查看 文件夹 dir(tf.data) help(tf.keras)