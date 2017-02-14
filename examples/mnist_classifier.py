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

def plot_hist(w,name):
    plt.figure()
    plt.hist(w.flatten(),100)
    plt.title(name)
    plt.xlabel("Weight")
    plt.ylabel("Frequency")

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
caffe_root = './'
np.random.seed(8773)

val_path  = 'examples/mnist/mnist_test_mlp_lmdb/'
# val_path  = 'examples/mnist/mnist_test_cnn_lmdb/'

#mu, sigma = 0, 0.03  # mean and standard deviation
#num1 = 500
#num2 = 25000
#num3 = 400000
#num4 = 5000
#s1 = np.random.normal(mu, sigma, num1)
#plot_hist(s1, "{} noise".format("s1"))
#plt.show()
#s2 = np.random.normal(mu, sigma, num2)
#s3 = np.random.normal(mu, sigma, num3)
#s4 = np.random.normal(mu, sigma, num4)

# GPU mode
# caffe.set_device(0)
# caffe.set_mode_gpu()
caffe.set_mode_cpu()

# layer_names = ["conv1","conv2","ip1","ip2"]
layer_names = ["ip1","ip2","ip3"]
# layer_names = ["ip1","ip2"]
mu, sigma = 0, 0.035  # mean and standard deviation
# src_model = caffe_root + 'examples/mnist/lenet_iter_60000.caffemodel'
# src_model = caffe_root + 'examples/mnist/mlp_500_300.caffemodel'
# src_model = caffe_root + 'examples/mnist/mlp_500_300_quan.caffemodel'
src_model = caffe_root + 'examples/mnist/lenet_iter_60000_0.991.caffemodel'
# src_model = caffe_root + 'examples/mnist/lenet_bias_iter_60000.caffemodel'
# src_model = caffe_root + 'examples/mnist/lenet_iter_60000_cnn.caffemodel'
# src_model = caffe_root + 'examples/mnist/lenet_mlp_iter_60000_0.9723.caffemodel'
# src_model = caffe_root + 'examples/mnist/0.0001_0.0_0.0_0.0_0.0_Fri_Jul_15_09-52-31_EDT_2016/lenet_cnn_bias_iter_60000.caffemodel'
# src_model = caffe_root + 'examples/mnist/lenet_bias_iter_60000_0.98_diff_constrain_bias_lr0.01_decay_0.0015_0.001_0.004.caffemodel'
# src_model = caffe_root + 'examples/mnist/0.008_0.0001_0.0_0.0_0.0_Fri_Jul__1_14-41-26_EDT_2016/lenet_cnn_bias_iter_30000.caffemodel'
# src_model = caffe_root + 'examples/mnist/lenet_cnn_iter_16457.caffemodel'
# src_model = caffe_root + 'examples/mnist/lenet_cnn_bias_iter_30000_0.9896.caffemodel'
#net = caffe.Net(caffe_root + 'examples/mnist/lenet_deploy_mlp.prototxt'

net = caffe.Net(
              # caffe_root + 'examples/mnist/lenet_deploy_cnn.prototxt',
              # caffe_root + 'examples/mnist/lenet_deploy_mlp_784_128_10.prototxt',
              caffe_root + 'examples/mnist/lenet_deploy_mlp.prototxt',
              # caffe_root + 'examples/mnist/lenet_deploy_cnn_noise.prototxt',
              #caffe_root + 'examples/mnist/lenet_iter_60000.caffemodel',
              # caffe_root + 'examples/mnist/lenet_iter_60000_cnn_0.9915.caffemodel',
              src_model,
              #caffe_root + 'examples/mnist/lenet_cnn_noise_iter_30000.caffemodel',
              #caffe_root + 'examples/mnist/mlp_500_300.caffemodel',
              #caffe_root + 'examples/mnist/mlp_64_32.caffemodel',
              caffe.TEST)
#for layer_name in ["conv2"]:
#    weights = net.params[layer_name][0].data
#    w_f = weights.flatten()
#    plot_hist(w_f, "{}".format(layer_name))
#plt.show()


# quan_pair={"ip1":[-0.07,0,0.07],
#            "ip2":[-0.09,0,0.09],
#            "ip3":[-0.25,0,0.25]}

# quan_pair={"ip1":[-0.07,0,0.07],
#            "ip2":[-0.09,0,0.09]
#            }

# quan_pair = {"conv1": [-0.36, 0, 0.36],
#             "conv2": [-0.07, 0, 0.07],
#             "ip1": [-0.02, 0, 0.02],
#             "ip2": [-0.18, 0, 0.18]}

step=0.29
quan_pair = {"ip1": [-step, 0, step, -0.08, 0.08, -0.05, 0.05, -0.12, 0.12],
             "ip2": [-step, 0, step, -0.08, 0.08, -0.05, 0.05, -0.12, 0.12 ],
             "ip3": [-step, 0, step, -0.08, 0.08, -0.05, 0.05, -0.12, 0.12]}

# quan_pair = {"conv1": [-0.07, 0, 0.07],
#              "conv2": [-0.07, 0, 0.07],
#              "ip1": [-0.07, 0, 0.07],
#              "ip2": [-0.07, 0, 0.07]}
#

# # quan_pair = {"ip1": [-step, 0, step],
# #              "ip2": [-step, 0, step]}

for layername in quan_pair.iterkeys():
    qua_list = quan_pair[layername]
    weights = net.params[layername][0].data
    w_shape = weights.shape
    w_f = weights.flatten()
    # plot_hist(w_f, "{} before quantification".format(layername))
    for idx, val in enumerate(w_f):
        # quantification
        d = abs(val - qua_list[0])
        idx_qua = qua_list[0]
        for list_val in qua_list:
            if abs(val - list_val) < d:
                d = abs(val - list_val)
                idx_qua = list_val
        w_f[idx] = idx_qua
    # plot_hist(w_f, "{} after quantification".format(layername))
    weights[:] = w_f.reshape(w_shape)
plt.show()

'''
#
#for layer_name in ["conv2"]:
#    weights = net.params[layer_name][0].data
#    w_shape = weights.shape
#    w_f = weights.flatten()
#    plot_hist(w_f, "{} before noise".format(layer_name))
#    for idx, val in enumerate(w_f):
#        w_f[idx] = w_f[idx]+s2[idx]
#    plot_hist(w_f, "{} after noise".format(layer_name))
#    weights[:] = w_f.reshape(w_shape)
#
#for layer_name in ["ip1"]:
#    weights = net.params[layer_name][0].data
#    w_shape = weights.shape
#    w_f = weights.flatten()
#    plot_hist(w_f, "{} before noise".format(layer_name))
#    for idx, val in enumerate(w_f):
#        w_f[idx] = w_f[idx]+s3[idx]
#    plot_hist(w_f, "{} after noise".format(layer_name))
#    weights[:] = w_f.reshape(w_shape)
#
#for layer_name in ["ip2"]:
#    weights = net.params[layer_name][0].data
#    w_shape = weights.shape
#    w_f = weights.flatten()
#    plot_hist(w_f, "{} before noise".format(layer_name))
#    for idx, val in enumerate(w_f):
#        w_f[idx] = w_f[idx]+s4[idx]
#    plot_hist(w_f, "{} after noise".format(layer_name))
#    weights[:] = w_f.reshape(w_shape)

# imp=0.065
# #step_ip3 = 0.61
# #qua_list = [-0.59, -0.31, -0.25, -step_ip3, 0, step_ip3, 0.25, 0.31, 0.59]
# #step_conv1 = 0.36
# step_conv1 = imp
# qua_list = [-step_conv1, 0, step_conv1]
# for layer_name in ["conv1"]:
#         #net.params.keys():
#     #["ip1", "ip2"]:
#     weights = net.params[layer_name][0].data
#     w_shape = weights.shape
#     w_f = weights.flatten()
#     plot_hist(w_f, "{} before quantification".format(layer_name))
#     for idx, val in enumerate(w_f):
#         # quantification
#         d = abs(val - qua_list[0])
#         idx_qua = qua_list[0]
#         for list_val in qua_list:
#             if abs(val-list_val)<d:
#                 d = abs(val-list_val)
#                 idx_qua = list_val
#         w_f[idx] = idx_qua
#     plot_hist(w_f, "{} after quantification".format(layer_name))
#     weights[:] = w_f.reshape(w_shape)
#
# # step_ip3 = 0.61
# # qua_list = [-0.59, -0.31, -0.25, -step_ip3, 0, step_ip3, 0.25, 0.31, 0.59]
# #step_conv2 = 0.07
# step_conv2 = imp
# qua_list = [-step_conv2, 0, step_conv2]
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
#     plot_hist(w_f, "{} after quantification".format(layer_name))
#     weights[:] = w_f.reshape(w_shape)
#
# # #step_ip2 = 0.06
# # #qua_list = [-0.09, -step_ip2, 0, step_ip2, 0.09]
# #step_ip2 = 0.18
# step_ip2 = imp
# qua_list = [-step_ip2, 0, step_ip2]
# for layer_name in ["ip2"]:
#         #net.params.keys():
#     #["ip1", "ip2"]:
#     weights = net.params[layer_name][0].data
#     w_shape = weights.shape
#     w_f = weights.flatten()
#     plot_hist(w_f, "{} before quantification".format(layer_name))
#     for idx, val in enumerate(w_f):
#         # quantification
#         d = abs(val - qua_list[0])
#         idx_qua = qua_list[0]
#         for list_val in qua_list:
#             if abs(val-list_val)<d:
#                 d = abs(val-list_val)
#                 idx_qua = list_val
#         w_f[idx] = idx_qua
#     plot_hist(w_f, "{} after quantification".format(layer_name))
#     weights[:] = w_f.reshape(w_shape)
# #
# #step_ip1 = 0.03
# #qua_list = [-step_ip1, -0.1, 0, 0.1, step_ip1]
# #step_ip1 = 0.02
# step_ip1 = imp
# qua_list = [-step_ip1, 0, step_ip1]
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
#     plot_hist(w_f, "{} after quantification".format(layer_name))
#     weights[:] = w_f.reshape(w_shape)




#linear quantification
#step = 0.025
#qsize = 2*step
#for layer_name in net.params.keys():
#        weights = net.params[layer_name][0].data
#        w_shape = weights.shape
#        w_f = weights.flatten()
#        plot_hist(w_f,"{} before quantification".format(layer_name))
#        for idx, val in enumerate(w_f):
#            #quantification
#            if abs(val)>4*step:
#                w_f[idx] = sign(val)*4*step
#            else:
#                w_f[idx] = sign(val) * round(abs(val)/qsize) * qsize
#        plot_hist(w_f, "{} after quantification".format(layer_name))
#        weights[:] = w_f.reshape(w_shape)
#plt.show()
#exit()

#quantification for layer1
#layer1_step1 = 0.02
#layer1_step2 = 0.05
#
#layer2_step1 = 0.03
#layer2_step2 = 0.05
#
#layer3_step1 = 0.05
#layer3_step2 = 0.1
#layer3_step3 = 0.2
#layer3_step4 = 0.4


#step = 1
#qsize = 2*step
#for layer_name in net.params.keys():
#        weights = net.params[layer_name][0].data
#        w_shape = weights.shape
#        w_f = weights.flatten()
#        plot_hist(w_f,"{} before quantification".format(layer_name))
#        for idx, val in enumerate(w_f):
#            w_f[idx] = sign(val)
#            #quantification
#            #if abs(val)>3*step:
#            #    w_f[idx] = sign(val)*3*step
#            #else:
#            #    w_f[idx] = sign(val) * (floor(abs(val)/qsize) * qsize+step)
#        plot_hist(w_f, "{} after quantification".format(layer_name))
#        weights[:] = w_f.reshape(w_shape)

#step_ip3 = 0.61
#qua_list = [-0.59, -0.31, -0.25, -step_ip3, 0, step_ip3, 0.25, 0.31, 0.59]
#step_ip3 = 0.25
#qua_list = [-step_ip3, 0, step_ip3]
#for layer_name in ["ip3"]:
        #net.params.keys():
    #["ip1", "ip2"]:
    #    weights = net.params[layer_name][0].data
    #    w_shape = weights.shape
    #    w_f = weights.flatten()
    #    plot_hist(w_f, "{} before quantification".format(layer_name))
    #    for idx, val in enumerate(w_f):
    #        # quantification
    #        d = abs(val - qua_list[0])
    #        idx_qua = qua_list[0]
    #        for list_val in qua_list:
    #           if abs(val-list_val)<d:
    #                d = abs(val-list_val)
    #                idx_qua = list_val
    #        w_f[idx] = idx_qua
    #    plot_hist(w_f, "{} after quantification".format(layer_name))
#   weights[:] = w_f.reshape(w_shape)

#step_ip2 = 0.06
#qua_list = [-0.09, -step_ip2, 0, step_ip2, 0.09]
#step_ip2 = 0.09
#qua_list = [-step_ip2, 0, step_ip2]
#for layer_name in ["ip2"]:
        #net.params.keys():
    #["ip1", "ip2"]:
    #    weights = net.params[layer_name][0].data
    #    w_shape = weights.shape
    #    w_f = weights.flatten()
    #    plot_hist(w_f, "{} before quantification".format(layer_name))
    #    for idx, val in enumerate(w_f):
        # quantification
    #        d = abs(val - qua_list[0])
    #        idx_qua = qua_list[0]
    #        for list_val in qua_list:
    #            if abs(val-list_val)<d:
    #                d = abs(val-list_val)
    #                idx_qua = list_val
    #        w_f[idx] = idx_qua
    #    plot_hist(w_f, "{} after quantification".format(layer_name))
#   weights[:] = w_f.reshape(w_shape)

#step_ip1 = 0.03
#qua_list = [-step_ip1, -0.1, 0, 0.1, step_ip1]
#step_ip1 = 0.07
#qua_list = [-step_ip1, 0, step_ip1]
#for layer_name in ["ip1"]:
    # net.params.keys():
    # ["ip1", "ip2"]:
    #    weights = net.params[layer_name][0].data
    #    w_shape = weights.shape
    #    w_f = weights.flatten()
    #    plot_hist(w_f, "{} before quantification".format(layer_name))
    #    for idx, val in enumerate(w_f):
        # quantification
    #        d = abs(val - qua_list[0])
    #        idx_qua = qua_list[0]
    #        for list_val in qua_list:
    #           if abs(val - list_val) < d:
    #                d = abs(val - list_val)
    #                idx_qua = list_val
    #        w_f[idx] = idx_qua
    #    plot_hist(w_f, "{} after quantification".format(layer_name))
#    weights[:] = w_f.reshape(w_shape)




# set net to batch size
#height = 8
#width = 8
#if height!=width:
#    warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)
'''
count = 0
correct_top1 = 0
correct_top5 = 0
labels_set = set()
lmdb_env = lmdb.open(val_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

#mean_blob = caffe.proto.caffe_pb2.BlobProto()
#mean_data = open( 'examples/cifar10/mean.binaryproto' , 'rb' ).read()
#mean_blob.ParseFromString(mean_data)
#pixel_mean = np.array( caffe.io.blobproto_to_array(mean_blob) ).mean(0)

weight_pair = {}
for layer_name in layer_names:
    weight_pair[layer_name] = np.copy(net.params[layer_name][0].data)

# noise once
# for layer_name in layer_names:
#     weights = net.params[layer_name][0].data
#     weights[:] = weight_pair[layer_name] + np.random.normal(mu, sigma, weights.shape)

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

    net.blobs['data'].data[image_count%batch_size] = image/float(255)

    ##noise after a iteration
    #for layer_name in layer_names:
    #    weights = net.params[layer_name][0].data
    #    weights[:] = weight_pair[layer_name] + np.random.normal(mu, sigma, weights.shape)

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
# print(step_ip1)
# plt.show()
#file_split = os.path.splitext(src_model)
#filepath = file_split[0]+'_deployed'+file_split[1]
filepath,filename = os.path.split(src_model)
filepath = caffe_root+"./examples/mnist"
net.save(filepath+"/lenet_deployed_mnist.caffemodel")
print(filepath)