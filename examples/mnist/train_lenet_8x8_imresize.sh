#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lenet_solver_8x8_imresize.prototxt 
# --weights=examples/mnist/mlp_64_32.caffemodel
