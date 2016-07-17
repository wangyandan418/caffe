#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_cnn.prototxt \
    --weights=examples/cifar10/cifar10_full_iter_300000_0.8212.caffemodel

# reduce learning rate by factor of 10
#$TOOLS/caffe train \
#    --solver=examples/cifar10/cifar10_full_solver_lr1_cnn.prototxt \
#    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5

# reduce learning rate by factor of 10
#$TOOLS/caffe train \
#    --solver=examples/cifar10/cifar10_full_solver_lr2_cnn.prototxt \
#    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
