set ldm=..\..\x64\Release\cifar10_sample.exe

:%ldm% --data_path ../../../dataset\cifar10 --learning_rate 0.01  --epochs 260 --minibatch_size 20 --plot 1000 ^
:--on_memory_rate 0.5 --augmentation_rate 0.3 --validation_rate 0.3 --test_sample 100 --out_of_core 1
: > log.txt

%ldm% --data_path ../../../dataset\cifar10 --learning_rate 0.01  --epochs 300 --minibatch_size 128 --plot 50 ^
--on_memory_rate 0.02 --augmentation_rate 0.5 --test_sample 200 --decay_iter 100 --save_iter 2  --out_of_core 1 > log.txt
