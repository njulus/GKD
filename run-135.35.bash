#!/bin/bash

sleep 1 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 0 --depth 40 --width 2 --lambd 1 &
sleep 10 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 0 --depth 40 --width 1 --lambd 1 &
sleep 20 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 0 --depth 16 --width 2 --lambd 1 &
sleep 30 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 0 --depth 16 --width 1 --lambd 1 &

sleep 40 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 10 --depth 40 --width 2 --lambd 1 &
sleep 50 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 10 --depth 40 --width 1 --lambd 1 &
sleep 60 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 10 --depth 16 --width 2 --lambd 1 &
sleep 70 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 10 --depth 16 --width 1 --lambd 1 &

sleep 80 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 20 --depth 40 --width 2 --lambd 1 &
sleep 90 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 20 --depth 40 --width 1 --lambd 1 &
sleep 100 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 20 --depth 16 --width 2 --lambd 1 &
sleep 110 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 20 --depth 16 --width 1 --lambd 1 &

sleep 120 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 30 --depth 40 --width 2 --lambd 1 &
sleep 130 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 30 --depth 40 --width 1 --lambd 1 &
sleep 140 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 30 --depth 16 --width 2 --lambd 1 &
sleep 150 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 30 --depth 16 --width 1 --lambd 1 &

sleep 160 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 40 --depth 40 --width 2 --lambd 1 &
sleep 170 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 40 --depth 40 --width 1 --lambd 1 &
sleep 180 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 40 --depth 16 --width 2 --lambd 1 &
sleep 190 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 40 --depth 16 --width 1 --lambd 1 &

sleep 200 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 50 --depth 40 --width 2 --lambd 1 &
sleep 210 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 50 --depth 40 --width 1 --lambd 1 &
sleep 220 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 50 --depth 16 --width 2 --lambd 1 &
sleep 230 && python main.py --data_name CIFAR-100 --n_classes 50 --n_new_classes 50 --depth 16 --width 1 --lambd 1