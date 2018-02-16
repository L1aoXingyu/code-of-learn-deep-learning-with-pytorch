#!/usr/bin/env bash

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

if [ ! -e ./dataset ]; then
    mkdir ./dataset
fi

tar -xf VOCtrainval_11-May-2012.tar -C ./dataset
rm VOCtrainval_11-May-2012.tar
