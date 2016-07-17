#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lenet_solver_cnn_noise.prototxt \
 --weights=examples/mnist/lenet.caffemodel
