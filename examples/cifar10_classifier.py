__author__ = 'pittnuts'
'''
Main script to run classification/test/prediction/evaluation
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import *
from PIL import Image
import caffe
import sys
import os
import lmdb
from caffe.proto import caffe_pb2
from pittnuts import *
from os import system
from caffe_apps import *
import time
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
caffe_root = './'

def plot_hist(w,name):
    plt.figure()
    plt.hist(w,100)
    plt.title(name)
    plt.xlabel("Weight")
    plt.ylabel("Frequency")

val_path  = 'examples/cifar10/cifar10_test_lmdb'

mu, sigma = 0, 0.005  # mean and standard deviation
num1 = 2400
num2 = 25600
num3 = 51200
num4 = 10240
s1 = np.random.normal(mu, sigma, num1)
s2 = np.random.normal(mu, sigma, num2)
s3 = np.random.normal(mu, sigma, num3)
s4 = np.random.normal(mu, sigma, num4)


# GPU mode
#caffe.set_device(0)
#caffe.set_mode_gpu()

caffe.set_mode_cpu()
# src_model = caffe_root + 'examples/cifar10/0.0005_0.002_0.0_0.0_0.0_Sat_Jun_11_11-26-01_EDT_2016/cifar10_full_iter_130000.caffemodel'
# src_model = caffe_root + 'examples/cifar10/cifar10_full_iter_300000_0.8212.caffemodel'
# src_model = caffe_root + 'examples/cifar10/cifar10_full_bias_iter_150000.caffemodel'
# src_model = caffe_root + 'examples/cifar10/cifar10_full_iter_150000.caffemodel'
src_model = caffe_root + 'examples/cifar10/cifar10_full_bias_iter_150000_1level_diff_value.caffemodel'
net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_full_cnn.prototxt',
#              caffe_root + 'examples/cifar10/cifar10_full_iter_300000_0.8212.caffemodel',
                src_model,
              #caffe_root + 'examples/cifar10/cifar10_full_iter_300000.caffemodel',
              caffe.TEST)

step = 0.4
# quan_pair = {"conv1": [-step, 0, step],
#              "conv2": [-step, 0, step],
#              "conv3": [-step, 0, step],
#              "ip1": [-step, 0, step]}

# quan_pair = {"conv1": [-0.12, 0, 0.12],
#              "conv2": [-0.08, 0, 0.08],
#              "conv3": [-0.02, 0, 0.02],
#              "ip1": [-0.008, 0, 0.008]}
#
# for layername in quan_pair.iterkeys():
#     qua_list = quan_pair[layername]
#     weights = net.params[layername][0].data
#     w_shape = weights.shape
#     w_f = weights.flatten()
#     plot_hist(w_f, "{} before quantification".format(layername))
#     for idx, val in enumerate(w_f):
#         # quantification
#         d = abs(val - qua_list[0])
#         idx_qua = qua_list[0]
#         for list_val in qua_list:
#             if abs(val - list_val) < d:
#                 d = abs(val - list_val)
#                 idx_qua = list_val
#         w_f[idx] = idx_qua
#     plot_hist(w_f, "{} after quantification".format(layername))
#     weights[:] = w_f.reshape(w_shape)
# plt.show()

# # imp=0.1
# #step_ip3 = 0.61
# #qua_list = [-0.59, -0.31, -0.25, -step_ip3, 0, step_ip3, 0.25, 0.31, 0.59]
# #step_conv1 = 0.36
# step_conv1 = 0.17
# qua_list = [-step_conv1,-0.05,-0.12, -0.29, 0, step_conv1, 0.05, 0.12, 0.29]
# for layer_name in ["conv1"]:
#         #net.params.keys():
#     #["ip1", "ip2"]:
#     weights = net.params[layer_name][0].data
#     w_shape = weights.shape
#     w_f = weights.flatten()
#     plot_hist(w_f, "{} before quantification".format(layer_name))
#     for idx, val in enumerate(w_f):
#
#         # quantification
#         d = abs(val - qua_list[0])
#         idx_qua = qua_list[0]
#         for list_val in qua_list:
#             if abs(val-list_val)<d:
#                 d = abs(val-list_val)
#                 idx_qua = list_val
#         w_f[idx] = idx_qua
#         # add variance
#         w_f[idx] = w_f[idx] + s1[idx]
#     plot_hist(w_f, "{} after quantification".format(layer_name))
#     weights[:] = w_f.reshape(w_shape)
#
# # step_ip3 = 0.61
# # qua_list = [-0.59, -0.31, -0.25, -step_ip3, 0, step_ip3, 0.25, 0.31, 0.59]
# #step_conv2 = 0.07
# step_conv2 = 0.02
# qua_list = [-step_conv2,-0.05, -0.08,-0.2, 0, step_conv2, 0.05, 0.08,0.2]
# for layer_name in ["conv2"]:
#     # net.params.keys():
#     # ["ip1", "ip2"]:
#     weights = net.params[layer_name][0].data
#     w_shape = weights.shape
#     w_f = weights.flatten()
#     plot_hist(w_f, "{} before quantification".format(layer_name))
#     for idx, val in enumerate(w_f):
#         # quantification
#         d = abs(val - qua_list[0])
#         idx_qua = qua_list[0]
#         for list_val in qua_list:
#             if abs(val - list_val) < d:
#                 d = abs(val - list_val)
#                 idx_qua = list_val
#         w_f[idx] = idx_qua
#         # add variance
#         w_f[idx] = w_f[idx] + s2[idx]
#     plot_hist(w_f, "{} after quantification".format(layer_name))
#     weights[:] = w_f.reshape(w_shape)
#
# # #step_ip2 = 0.06
# # #qua_list = [-0.09, -step_ip2, 0, step_ip2, 0.09]
# #step_ip2 = 0.18
# step_conv3 = 0.01
# qua_list = [-step_conv3, -0.02, -0.08,-0.18, 0, step_conv3, 0.02, 0.08, 0.18]
# for layer_name in ["conv3"]:
#         #net.params.keys():
#     #["ip1", "ip2"]:
#     weights = net.params[layer_name][0].data
#     w_shape = weights.shape
#     w_f = weights.flatten()
#     plot_hist(w_f, "{} before quantification".format(layer_name))
#     for idx, val in enumerate(w_f):
#         #quantification
#         d = abs(val - qua_list[0])
#         idx_qua = qua_list[0]
#         for list_val in qua_list:
#             if abs(val-list_val)<d:
#                 d = abs(val-list_val)
#                 idx_qua = list_val
#         w_f[idx] = idx_qua
#         # add variance
#         w_f[idx] = w_f[idx] + s3[idx]
#     plot_hist(w_f, "{} after quantification".format(layer_name))
#     weights[:] = w_f.reshape(w_shape)
#
# #step_ip1 = 0.03
# #qua_list = [-step_ip1, -0.1, 0, 0.1, step_ip1]
# #step_ip1 = 0.02
# step_ip1 = 0.003
# qua_list = [-step_ip1,-0.008, -0.001, -0.012, 0, step_ip1, 0.008, 0.001, 0.012]
# for layer_name in ["ip1"]:
#     # net.params.keys():
#     # ["ip1", "ip2"]:
#     weights = net.params[layer_name][0].data
#     w_shape = weights.shape
#     w_f = weights.flatten()
#     plot_hist(w_f, "{} before quantification".format(layer_name))
#     for idx, val in enumerate(w_f):
#         # quantification
#         d = abs(val - qua_list[0])
#         idx_qua = qua_list[0]
#         for list_val in qua_list:
#             if abs(val - list_val) < d:
#                 d = abs(val - list_val)
#                 idx_qua = list_val
#         w_f[idx] = idx_qua
#         # add variance
#         w_f[idx] = w_f[idx] + s4[idx]
#     plot_hist(w_f, "{} after quantification".format(layer_name))
#     weights[:] = w_f.reshape(w_shape)

# set net to batch size
height = 32
width = 32
if height!=width:
    warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)

count = 0
correct_top1 = 0
correct_top5 = 0
labels_set = set()
lmdb_env = lmdb.open(val_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

#pixel_mean = np.load(caffe_root + 'examples/cifar10/mean.binaryproto').mean(1).mean(1)
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_data = open( 'examples/cifar10/mean.binaryproto' , 'rb' ).read()
mean_blob.ParseFromString(mean_data)
#pixel_mean = np.array( caffe.io.blobproto_to_array(mean_blob) ).mean(2).mean(2)
#pixel_mean = tile(pixel_mean.reshape([1,3]),(height*width,1)).reshape(height,width,3).transpose(2,0,1)

pixel_mean = np.array( caffe.io.blobproto_to_array(mean_blob) ).mean(0)

avg_time = 0
batch_size = net.blobs['data'].num
label = zeros((batch_size,1))
image_count = 0
for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label[image_count%batch_size,0] = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)

    #out = net.forward_all(data=np.asarray([image]))
    #image_tmp = image[(0,1,2),:,:]
    #image_tmp = image_tmp.transpose(0,2,1)
    #plt.imshow(image.transpose(1,2,0)[:,:,(2,1,0)])
    #plt.show()

    #crop_range = range(14,14+227)
    #image = image[:,14:14+227,14:14+227]

    net.blobs['data'].data[image_count%batch_size] = image-pixel_mean
    if image_count % batch_size == (batch_size-1):
        starttime = time.time()
        out = net.forward()
        endtime = time.time()
        plabel = out['prob'][:].argmax(axis=1)
        plabel_top5 = argsort(out['prob'][:],axis=1)[:,-1:-6:-1]
        assert (plabel==plabel_top5[:,0]).all()
        count = image_count + 1
        current_test_time = endtime-starttime

        correct_top1 = correct_top1 + sum(label.flatten() == plabel.flatten())#(1 if iscorrect else 0)

        correct_top5_count = sum(contains2D(plabel_top5,label))
        correct_top5 = correct_top5 + correct_top5_count

        sys.stdout.write("\n[{}] Accuracy (Top 1): {:.2f}%".format(count,100.*correct_top1/count))
        sys.stdout.write("  (Top 5): %.2f%%" % (100.*correct_top5/count))
        sys.stdout.write("  (current time): {}\n".format(1000*current_test_time))
        sys.stdout.flush()
    image_count += 1

print(step)
# plt.show()
# file_split = os.path.splitext(src_model)
# filepath = file_split[0]+'_q'+file_split[1]
# net.save(filepath)
filepath,filename = os.path.split(src_model)
net.save(filepath+"/lenet_deployed.caffemodel")