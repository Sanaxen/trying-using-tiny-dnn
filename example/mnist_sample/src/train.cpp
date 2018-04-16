/*
	Copyright (c) 2013, Taiga Nomi and the respective contributors
	All rights reserved.

	Use of this source code is governed by a BSD-style license that can be found
	in the LICENSE file.
	
	Copyright (c) 2018, Sanaxn
	All rights reserved.

	Use of this source code is governed by a BSD-style license that can be found
	in the LICENSE file.
	*/
#include <iostream>
#include "dnn_util.hpp"
#include "mnist_loader.hpp"

#define IMAGE_PADDING	2
//#define USE_ORG_NET

static void construct_net(tiny_dnn::network2<tiny_dnn::sequential> &nn,
	tiny_dnn::core::backend_t backend_type) {

	using tanh = tiny_dnn::activation::tanh;

#define O true
#define X false
	// clang-format off
	static const bool tbl[] = {
		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
	};
	// clang-format on
#undef O
#undef X

	size_t in_w = MNIST_IMAGE_WIDTH + IMAGE_PADDING * 2;
	size_t in_h = MNIST_IMAGE_HEIGHT + IMAGE_PADDING * 2;
	size_t in_map = MNIST_IMAGE_CHANNEL;

#ifdef USE_ORG_NET
	LayerInfo layers(in_w, in_h, in_map);
	nn << layers.add_cnv(6, 5, 1, padding::valid);
	nn << layers.tanh();
	nn << layers.add_avepool(2, 2);
	nn << layers.tanh();
	nn << layers.add_cnv(16, 5, 1, padding::valid, true, connection_table(tbl, 6, 16));
	nn << layers.tanh();
	nn << layers.add_avepool(2, 2);
	nn << layers.tanh();
	nn << layers.add_cnv(120, 5, 1, padding::valid);
	nn << layers.tanh();
	nn << layers.add_fc(10);
	nn << layers.tanh();;
#else
	LayerInfo layers(in_w, in_h, in_map, backend_type);

	nn << layers.add_cnv(32, 3, 1, padding::valid);
	nn << layers.relu();
	nn << layers.add_batnorm();
	nn << layers.add_cnv(32, 3, 1, padding::valid);
	nn << layers.relu();
	nn << layers.add_maxpool(2, 2);
	nn << layers.add_dropout(0.5);

	nn << layers.add_cnv(64, 3, 1, padding::valid);
	nn << layers.relu();
	nn << layers.add_batnorm();
	nn << layers.add_cnv(64, 3, 1, padding::valid);
	nn << layers.relu();
	nn << layers.add_maxpool(2, 2);
	nn << layers.add_dropout(0.5);

	nn << layers.add_fc(256);
	nn << layers.relu();
	nn << layers.add_dropout(0.5);
	nn << layers.add_fc(10);
	nn << tiny_dnn::activation::softmax(10);
#endif

	printf("layers:%zd\n", nn.depth());
	//for (int i = 0; i < nn.depth(); i++) {
	   // std::cout << "#layer:" << i << "\n";
	   // std::cout << "layer type:" << nn[i]->layer_type() << "\n";
	   // std::cout << "input:" << nn[i]->in_data_size() << "(" << nn[i]->in_data_shape() << ")\n";
	   // std::cout << "output:" << nn[i]->out_data_size() << "(" << nn[i]->out_data_shape() <<")\n";
	//}

	// generate graph model in dot language
	std::ofstream ofs("graph_net_mnist.txt");
	tiny_dnn::graph_visualizer viz(nn, "graph");
	viz.generate(ofs);
	printf("dot -Tgif graph_net_mnist.txt -o graph_mnist.gif\n");
}

size_t data_gen(DNNParameter& prm, ImageWorkBufferList& train_buf, ImageWorkBufferList& test_buf)
{

	const size_t train_num = (size_t)((MNIST_IMAGE_TRAIN_NUM*prm.on_memory_rate) / prm.n_minibatch)*prm.n_minibatch;
	const size_t augment_num = (size_t)((MNIST_IMAGE_TRAIN_NUM*prm.augmentation_rate*prm.on_memory_rate) / prm.n_minibatch)*prm.n_minibatch;

	const size_t train_num_max = train_num + augment_num;
	printf("train num   = %zd\n", train_num);
	printf("augment_num = %zd\n", augment_num);
	printf("train num+augment_num = %zd\n", train_num_max);

	LoadMinist(prm.data_dir_path, train_buf, test_buf, 0, MNIST_IMAGE_TRAIN_NUM, MNIST_IMAGE_TRAIN_NUM*prm.augmentation_rate);

	return train_num_max;
}


static void train_lenet(DNNParameter& prm) {
	// specify loss-function and learning strategy
	tiny_dnn::network2<tiny_dnn::sequential> nn;
#ifdef USE_ORG_NET
	tiny_dnn::adagrad optimizer;
	std::cout << "optimizer:" << "adagrad" << std::endl;
#else
	tiny_dnn::adam optimizer;
	std::cout << "optimizer:" << "adam" << std::endl;
#endif

	construct_net(nn, prm.backend_type);

	std::cout << "load models..." << std::endl;

	// load MNIST dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	ImageWorkBufferList train_buf(MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, prm.data_dir_path);
	ImageWorkBufferList test_buf(MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, prm.data_dir_path);

	size_t train_num_max = data_gen(prm, train_buf, test_buf);
	printf("all train %zd -> %zd\n", train_buf.data_num, train_num_max);
	nn.set_input_size(train_buf.data_num);

	train_images.resize(train_num_max);
	train_labels.resize(train_num_max);
	test_images.resize(MNIST_IMAGE_TEST_NUM);
	test_labels.resize(MNIST_IMAGE_TEST_NUM);
	set_image_data(train_buf, train_labels, train_images, 0, MNIST_IMAGE_CHANNEL, IMAGE_PADDING);
	set_image_data(test_buf, test_labels, test_images, 0, MNIST_IMAGE_CHANNEL, IMAGE_PADDING);
	printf("train_images %zd\n", train_images.size());

	train_buf.clear();
	test_buf.clear();

	printf("training total %zd set\n", train_images.size());
	printf("test total %zd set\n", test_images.size());

	std::cout << "start training" << std::endl;

	optimizer.alpha *=
		std::min(tiny_dnn::float_t(4),
			static_cast<tiny_dnn::float_t>(sqrt(prm.n_minibatch) * prm.learning_rate));
	std::cout << "optimizer.alpha:" << optimizer.alpha << std::endl;

	tiny_dnn::progress_display disp(nn.get_input_size());
	tiny_dnn::timer t;

	std::vector<tiny_dnn::tensor_t> inputs;
	std::vector<tiny_dnn::tensor_t> desired_outputs;

	nn.normalize_tensor(train_images, inputs);
	nn.normalize_tensor(train_labels, desired_outputs);

#ifdef USE_ORG_NET
	using train_loss = tiny_dnn::mse;
#else
	using train_loss = tiny_dnn::cross_entropy_multiclass;
#endif

	FILE* fp_accuracy_rate = NULL;
	FILE* fp_error_loss = NULL;
	if (prm.plot)
	{
		fp_accuracy_rate = fopen("accuracy_rate.dat", "w");
		fp_error_loss = fopen("error_loss.dat", "w");
	}

	if (prm.test_sample > train_images.size())
	{
		printf("error test_sample > train_images.size()\n");
		return;
	}
	std::vector<tiny_dnn::label_t> vari_labels(prm.test_sample);
	std::vector<tiny_dnn::vec_t> vari_images(prm.test_sample);
	for (int i = 0; i < prm.test_sample; i++)
	{
		vari_images[i] = train_images[i];
		vari_labels[i] = train_labels[i];
	}

	int load_count = 1;
	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "Epoch " << epoch << "/" << prm.n_train_epochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;
		//tiny_dnn::result res = nn.test(test_images, test_labels);
		
		std::mt19937 mt(epoch);
		std::uniform_int_distribution<int> rand_ts(0, (int)test_images.size() - 1);
		size_t sz = std::min(prm.test_sample, test_images.size());
		std::vector<tiny_dnn::label_t> tmp_labels(sz);
		std::vector<tiny_dnn::vec_t> tmp_images(sz);

#pragma omp parallel for
		for (int i = 0; i < sz; i++)
		{
			const int idx = rand_ts(mt);
			tmp_labels[i] = test_labels[idx];
			tmp_images[i] = test_images[idx];
		}
		tiny_dnn::result res = nn.test(tmp_images, tmp_labels);
		std::cout << res.num_success << "/" << res.num_total << std::endl;
		printf("%.3f%%\n", res.accuracy());

		std::vector<tiny_dnn::tensor_t> in;
		std::vector<tiny_dnn::tensor_t> te;
		nn.normalize_tensor(train_images, in);
		nn.normalize_tensor(train_labels, te);

		float_t loss = nn.get_loss<train_loss>(in, te);
		printf("loss:%.3f\n", loss);

		disp.restart(nn.get_input_size());
		t.restart();
		nn.set_netphase(tiny_dnn::net_phase::train);


		if (epoch % prm.decay_iter == 0)
		{
			optimizer.alpha *= float_t(0.1);
			std::cout << "optimizer.alpha:" << optimizer.alpha << std::endl;
		}
		if (epoch % prm.save_iter == 0)
		{
			// test and show results
			tiny_dnn::result res = nn.test(test_images, test_labels);

			char fname[512];
			sprintf(fname, "%s_%.3f%%.tmp", "mnist-model", res.accuracy());
			// save network model & trained weights
			nn.save(fname);
			nn.set_netphase(tiny_dnn::net_phase::train);
		}
	};


	int count = 0;
	auto on_enumerate_minibatch = [&]() {
		disp += prm.n_minibatch;
		count++;

		if (prm.plot)
		{
			if (count && count % prm.plot == 0)
			{
				std::mt19937 mt(count);
				std::uniform_int_distribution<int> rand_tr(0, (int)vari_images.size() - 1);
				std::uniform_int_distribution<int> rand_ts(0, (int)test_images.size() - 1);

				int sz = std::min(prm.test_sample, train_images.size()-1);

				std::vector<tiny_dnn::label_t> tmp_labels(sz);
				std::vector<tiny_dnn::vec_t> tmp_images(sz);

#pragma omp parallel for
				for (int i = 0; i < sz; i++)
				{
					const int idx = rand_tr(mt);
					tmp_labels[i] = vari_labels[idx];
					tmp_images[i] = vari_images[idx];
				}
				tiny_dnn::result res = nn.test(tmp_images, tmp_labels);
				float_t validation_acc = res.accuracy();
				float_t validation_error = 0;// nn.get_loss<train_loss>(tmp_images, tmp_labels);

				std::vector<tiny_dnn::tensor_t> in;
				std::vector<tiny_dnn::tensor_t> te;
				nn.normalize_tensor(tmp_images, in);
				nn.normalize_tensor(tmp_labels, te);

				validation_error = nn.get_loss<train_loss>(in, te);

				sz = std::min(prm.test_sample, test_images.size());
				tmp_labels.resize(sz);
				tmp_images.resize(sz);
#pragma omp parallel for
				for (int i = 0; i < sz; i++)
				{
					const int idx = rand_ts(mt);
					tmp_labels[i] = test_labels[idx];
					tmp_images[i] = test_images[idx];
				}
				res = nn.test(tmp_images, tmp_labels);
				float_t test_acc = res.accuracy();
				float_t test_error = 0;// nn.get_loss<train_loss>(tmp_images, tmp_labels);
				
				nn.normalize_tensor(tmp_images, in);
				nn.normalize_tensor(tmp_labels, te);

				test_error = nn.get_loss<train_loss>(in, te);

				nn.set_netphase(tiny_dnn::net_phase::train);

				if (fp_accuracy_rate)
				{
					fprintf(fp_accuracy_rate, "%d %.3f %.3f\n", count, validation_acc, test_acc);
					fflush(fp_accuracy_rate);
				}
				if (fp_error_loss)
				{
					fprintf(fp_error_loss, "%d %.3f %.3f\n", count, validation_error, test_error);
					fflush(fp_error_loss);
				}
			}
		}
	};


	auto load_tensor_data = [&]() {
		//std::cout << "Next training data load..." << std::endl;
		set_image_data(train_buf, train_labels, train_images, load_count, MNIST_IMAGE_CHANNEL, IMAGE_PADDING);
		train_buf.clear();
		test_buf.clear();

		nn.normalize_tensor(train_images, inputs);
		nn.normalize_tensor(train_labels, desired_outputs);
		load_count++;
	};

	// training
	nn.fit2<train_loss>(optimizer, inputs, desired_outputs,
		prm.n_minibatch,
		prm.n_train_epochs, 
		on_enumerate_minibatch, 
		on_enumerate_epoch,
		load_tensor_data);

	std::cout << "end training." << std::endl;

	if (fp_accuracy_rate) fclose(fp_accuracy_rate);
	if (fp_error_loss)fclose(fp_error_loss);

	// test and show results
	tiny_dnn::result res = nn.test(test_images, test_labels);

	//ConfusionMatrix
	std::cout << "ConfusionMatrix:" << std::endl;
	res.print_detail(std::cout);
	std::cout << res.num_success << "/" << res.num_total << std::endl;
	printf("%.3f%%\n", res.accuracy());
	// save network model & trained weights
	nn.save("mnist-model");
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
	const std::array<const std::string, 5> names = { {
	  "internal", "nnpack", "libdnn", "avx", "opencl",
	} };
	for (size_t i = 0; i < names.size(); ++i) {
		if (name.compare(names[i]) == 0) {
			return static_cast<tiny_dnn::core::backend_t>(i);
		}
	}
	return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
	std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
		<< " --learning_rate 1"
		<< " --epochs 30"
		<< " --minibatch_size 16"
		<< " --backend_type internal"
		<< " --on_memory_rate 0.8"
		<< " --augmentation_rate 0.2"
		<< " --test_sample 100"
		<< " --decay_iter 100"
		<< " --out_of_core 0"
		<< " --plot 0" << std::endl;
}

int main(int argc, char **argv) {
	DNNParameter prm;
	prm.learning_rate = 1;
	prm.n_train_epochs = 30;
	prm.data_dir_path = "";
	prm.n_minibatch = 16;
	prm.backend_type = tiny_dnn::core::default_engine();
	prm.plot = 0;
	prm.on_memory_rate = 0.8f;
	prm.augmentation_rate = 0.2f;
	prm.test_sample = 100;
	prm.out_of_core = false;
	prm.decay_iter = 100;

	if (argc == 2) {
		std::string argname(argv[1]);
		if (argname == "--help" || argname == "-h") {
			usage(argv[0]);
			return 0;
		}
	}
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--learning_rate") {
			prm.learning_rate = atof(argv[count + 1]);
		}
		else if (argname == "--epochs") {
			prm.n_train_epochs = atoi(argv[count + 1]);
		}
		else if (argname == "--minibatch_size") {
			prm.n_minibatch = atoi(argv[count + 1]);
		}
		else if (argname == "--backend_type") {
			prm.backend_type = parse_backend_name(argv[count + 1]);
		}
		else if (argname == "--data_path") {
			prm.data_dir_path = std::string(argv[count + 1]);
		}
		else if (argname == "--on_memory_rate") {
			prm.on_memory_rate = atof(argv[count + 1]);
		}
		else if (argname == "--augmentation_rate") {
			prm.augmentation_rate = atof(argv[count + 1]);
		}
		else if (argname == "--test_sample") {
			prm.test_sample = atoi(argv[count + 1]);
		}
		else if (argname == "--plot") {
			prm.plot = atoi(argv[count + 1]);
		}
		else if (argname == "--out_of_core") {
			prm.out_of_core = atoi(argv[count + 1]) != 0 ? true : false;
		}
		else if (argname == "--decay_iter") {
			prm.decay_iter = atoi(argv[count + 1]);
		}
		else if (argname == "--save_iter") {
			prm.save_iter = atoi(argv[count + 1]);
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			usage(argv[0]);
			return -1;
		}

	}
	if (prm.data_dir_path == "") {
		std::cerr << "Data path not specified." << std::endl;
		usage(argv[0]);
		return -1;
	}
	if (prm.learning_rate <= 0) {
		std::cerr
			<< "Invalid learning rate. The learning rate must be greater than 0."
			<< std::endl;
		return -1;
	}
	if (prm.n_train_epochs <= 0) {
		std::cerr << "Invalid number of epochs. The number of epochs must be "
			"greater than 0."
			<< std::endl;
		return -1;
	}
	if (prm.n_minibatch <= 0 || prm.n_minibatch > 60000) {
		std::cerr
			<< "Invalid minibatch size. The minibatch size must be greater than 0"
			" and less than dataset size (60000)."
			<< std::endl;
		return -1;
	}
	std::cout << "Running with the following parameters:" << std::endl
		<< "Data path: " << prm.data_dir_path << std::endl
		<< "Learning rate: " << prm.learning_rate << std::endl
		<< "Minibatch size: " << prm.n_minibatch << std::endl
		<< "Number of epochs: " << prm.n_train_epochs << std::endl
		<< "Backend type: " << prm.backend_type << std::endl
		<< "On memory rate: " << prm.on_memory_rate << "%" << std::endl
		<< "Augmentation rate : " << prm.augmentation_rate << "%" << std::endl
		<< "Test sampling   : " << prm.test_sample << std::endl
		<< "Out of core training data   : " << prm.out_of_core << std::endl
		<< "Accuracy plotting cycle : " << prm.plot << std::endl
		<< std::endl;
	try {
		train_lenet(prm);
	}
	catch (tiny_dnn::nn_error &err) {
		std::cerr << "Exception: " << err.what() << std::endl;
	}
	return 0;
}
