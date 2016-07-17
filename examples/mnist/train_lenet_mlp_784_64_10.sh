#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lenet_solver_mlp_784_64_10.prototxt \
# --weights=examples/mnist/mlp_500_300.caffemodel

