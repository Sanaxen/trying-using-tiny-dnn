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
#include "cifar10_loader.hpp"

#define IMAGE_PADDING	0
//#define USE_ORG_NET


size_t data_gen(DNNParameter& prm, ImageWorkBufferList& train_buf, ImageWorkBufferList& test_buf)
{
	const int train_num = (int)((CIFAR10_IMAGE_TRAIN_NUM*prm.on_memory_rate) / prm.n_minibatch)*prm.n_minibatch;
	const int augment_num = (int)((CIFAR10_IMAGE_TRAIN_NUM*prm.augmentation_rate*prm.on_memory_rate) / prm.n_minibatch)*prm.n_minibatch;

	const int train_num_max = train_num + augment_num;
	printf("train num   = %d\n", train_num);
	printf("augment_num = %d\n", augment_num);
	printf("train num+augment_num = %d\n", train_num_max);

	LoadCifar10(prm.data_dir_path, train_buf, test_buf, 0, CIFAR10_IMAGE_TRAIN_NUM, CIFAR10_IMAGE_TRAIN_NUM*prm.augmentation_rate);

	return train_num_max;
}

static void test_net(DNNParameter& prm) {
	// specify loss-function and learning strategy
	tiny_dnn::network2<tiny_dnn::sequential> nn;

	nn.load("cifar10-model");
	std::cout << "load models..." << std::endl;

	// load CIFAR10 dataset
	std::vector<tiny_dnn::label_t> test_labels;
	std::vector<tiny_dnn::vec_t>  test_images;

	ImageWorkBufferList train_buf(CIFAR10_IMAGE_WIDTH, CIFAR10_IMAGE_HEIGHT, prm.data_dir_path);
	ImageWorkBufferList test_buf(CIFAR10_IMAGE_WIDTH, CIFAR10_IMAGE_HEIGHT, prm.data_dir_path);

	size_t train_num_max = data_gen(prm, train_buf, test_buf);
	printf("all train %zd -> %zd\n", train_buf.data_num, train_num_max);
	nn.set_input_size(train_buf.data_num);


	test_images.resize(CIFAR10_IMAGE_TEST_NUM);
	test_labels.resize(CIFAR10_IMAGE_TEST_NUM);
	set_image_data(test_buf, test_labels, test_images, 0, CIFAR10_IMAGE_CHANNEL, IMAGE_PADDING);

	test_buf.clear();

	printf("test_images:%zd\n", test_images.size());
	printf("test_labels:%zd\n", test_labels.size());
	fflush(stdout);

	// test and show results
	tiny_dnn::result res = nn.test(test_images, test_labels);
	
	//ConfusionMatrix
	std::cout << "ConfusionMatrix:" << std::endl;
	res.print_detail(std::cout);
	std::cout << res.num_success << "/" << res.num_total << std::endl;
	printf("%.3f%%\n", res.accuracy());
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
	std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder" << std::endl;
}


int main(int argc, char **argv) {
	DNNParameter prm;
	prm.data_dir_path = "";
	prm.backend_type = tiny_dnn::core::default_engine();
	prm.out_of_core = false;

	if (argc == 2) {
		std::string argname(argv[1]);
		if (argname == "--help" || argname == "-h") {
			usage(argv[0]);
			return 0;
		}
	}
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--data_path") {
			prm.data_dir_path = std::string(argv[count + 1]);
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
	try {
		test_net(prm);
	}
	catch (tiny_dnn::nn_error &err) {
		std::cerr << "Exception: " << err.what() << std::endl;
	}
	return 0;
}
