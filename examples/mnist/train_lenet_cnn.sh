#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lenet_solver_cnn.prototxt \
-gpu 1 \
--weights=examples/mnist/lenet_iter_60000_cnn.caffemodel
