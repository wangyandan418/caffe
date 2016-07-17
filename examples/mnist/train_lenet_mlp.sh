#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lenet_solver_mlp.prototxt \
 --weights=examples/mnist/mlp_500_300.caffemodel

