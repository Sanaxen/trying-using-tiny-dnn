:set ldm=..\..\x64\Release\example_mnist_train_org.exe
:%ldm% --data_path ../../../dataset\mnist

set ldm=..\..\x64\Release\mnist_sample.exe


%ldm% --data_path ../../../dataset\mnist --learning_rate 0.005  --minibatch_size 30 --epochs 600 ^
--plot 100 --on_memory_rate 0.1 --augmentation_rate 0.0005 --test_sample 1000 --decay_iter 200 --save_iter 2 --out_of_core 1 > log.txt
