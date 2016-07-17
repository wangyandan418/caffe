#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lenet_solver_mlp_bias.prototxt \
 --weights=examples/mnist/lenet_deployed_mnist.caffemodel
