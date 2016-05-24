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
import lmdb
from caffe.proto import caffe_pb2
from pittnuts import *
from os import system
from caffe_apps import *
import time

def plot_hist(w,name):
    plt.figure()
    plt.hist(w,100)
    plt.title(name)
    plt.xlabel("Weight")
    plt.ylabel("Frequency")

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
caffe_root = './'

val_path  = 'examples/mnist/mnist_test_mlp_lmdb/'

# GPU mode
caffe.set_device(1)
caffe.set_mode_gpu()

#caffe.set_mode_cpu()

net = caffe.Net(caffe_root + 'examples/mnist/lenet_deploy_mlp.prototxt',
              #caffe_root + 'examples/mnist/lenet_iter_60000.caffemodel',
                caffe_root + 'examples/mnist/mlp_500_300.caffemodel',
              #caffe_root + 'examples/mnist/mlp_64_32.caffemodel',
              caffe.TEST)

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

step_ip3 = 0.61
qua_list = [-0.59, -0.31, -0.25, -step_ip3, 0, step_ip3, 0.25, 0.31, 0.59]
#qua_list = [-step, 0, step]
for layer_name in ["ip3"]:
        #net.params.keys():
    #["ip1", "ip2"]:
    weights = net.params[layer_name][0].data
    w_shape = weights.shape
    w_f = weights.flatten()
    plot_hist(w_f, "{} before quantification".format(layer_name))
    for idx, val in enumerate(w_f):
        # quantification
        d = abs(val - qua_list[0])
        idx_qua = qua_list[0]
        for list_val in qua_list:
            if abs(val-list_val)<d:
                d = abs(val-list_val)
                idx_qua = list_val
        w_f[idx] = idx_qua
    plot_hist(w_f, "{} after quantification".format(layer_name))
    weights[:] = w_f.reshape(w_shape)

step_ip2 = 0.06
qua_list = [-0.09, -step_ip2, 0, step_ip2, 0.09]
#qua_list = [-0.09, 0, 0.09]
for layer_name in ["ip2"]:
        #net.params.keys():
    #["ip1", "ip2"]:
    weights = net.params[layer_name][0].data
    w_shape = weights.shape
    w_f = weights.flatten()
    plot_hist(w_f, "{} before quantification".format(layer_name))
    for idx, val in enumerate(w_f):
        # quantification
        d = abs(val - qua_list[0])
        idx_qua = qua_list[0]
        for list_val in qua_list:
            if abs(val-list_val)<d:
                d = abs(val-list_val)
                idx_qua = list_val
        w_f[idx] = idx_qua
    plot_hist(w_f, "{} after quantification".format(layer_name))
    weights[:] = w_f.reshape(w_shape)

step_ip1 = 0.03
qua_list = [-step_ip1, -0.1, 0, 0.1, step_ip1]
for layer_name in ["ip1"]:
    # net.params.keys():
    # ["ip1", "ip2"]:
    weights = net.params[layer_name][0].data
    w_shape = weights.shape
    w_f = weights.flatten()
    plot_hist(w_f, "{} before quantification".format(layer_name))
    for idx, val in enumerate(w_f):
        # quantification
        d = abs(val - qua_list[0])
        idx_qua = qua_list[0]
        for list_val in qua_list:
            if abs(val - list_val) < d:
                d = abs(val - list_val)
                idx_qua = list_val
        w_f[idx] = idx_qua
    plot_hist(w_f, "{} after quantification".format(layer_name))
    weights[:] = w_f.reshape(w_shape)




# set net to batch size
height = 8
width = 8
if height!=width:
    warnings.warn("height!=width, please double check their dimension position",RuntimeWarning)

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

print(step_ip2)
plt.show()
