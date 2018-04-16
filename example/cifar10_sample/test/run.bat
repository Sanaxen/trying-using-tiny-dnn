set ldm=..\..\x64\Release\cifar10_sample.exe

%ldm% --data_path ../../../dataset\cifar10 --learning_rate 0.003  --minibatch_size 15 --epochs 290 ^
--plot 1000 --on_memory_rate 0.005 --augmentation_rate 0.1 --test_sample 100 --out_of_core 1 > log.txt
