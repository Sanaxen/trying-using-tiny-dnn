/*
Copyright (c) 2018, Sanaxn
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef _MNIST_LOADER_HPP

#define _MNIST_LOADER_HPP

#define MNIST_IMAGE_CHANNEL	1	
#define MNIST_IMAGE_WIDTH	28
#define MNIST_IMAGE_HEIGHT	28	
#define MNIST_IMAGE_SIZE	(MNIST_IMAGE_WIDTH*MNIST_IMAGE_HEIGHT)
#define MNIST_IMAGE_TRAIN_NUM		60000
#define MNIST_IMAGE_TEST_NUM		10000

#include "image/ImageWorkBuffer.hpp"

inline int LoadMinist(std::string dataDir, ImageWorkBufferList& train_buf, ImageWorkBufferList& test_buf, const size_t load_count, const int train_num, const int augment_num)
{
	std::mt19937 mt;
	std::uniform_real_distribution<double> d_rand;
	d_rand = std::uniform_real_distribution<double>(0.0, 1.0);

	std::vector<std::string> train;
	std::vector<std::string> test;

	std::vector<unsigned char> tmp(MNIST_IMAGE_SIZE * 3);


	std::string& train_filelist = std::string("train_list.txt");
	std::string& test_filelist = std::string("test_list.txt");
	
	train_buf.check_DiskOutCompleted(train_filelist);
	test_buf.check_DiskOutCompleted(test_filelist);

	img_contrast contrast;

	int numAugmentation = 0;
	int cnt = 0;

	if (!train_buf.disk_out_completed)
	{
		// read a test data of MNIST(http://yann.lecun.com/exdb/mnist/).
		const int N = MNIST_IMAGE_TRAIN_NUM;
		std::ifstream train_image(dataDir + "\\train-images.idx3-ubyte", ios_base::binary);
		if (!train_image.is_open()) {
			cerr << "\"train-images.idx3-ubyte\" is not found!" << endl;
			return -1;
		}
		std::ifstream train_label(dataDir + "\\train-labels.idx1-ubyte", ios_base::binary);
		if (!train_label.is_open()) {
			cerr << "\"train-labels.idx1-ubyte\" is not found!" << endl;
			return -1;
		}

		train_image.seekg(4 * 4, std::ios_base::beg);
		train_label.seekg(4 * 2, std::ios_base::beg);

		for (int i = 0; i < N; ++i)
		{
			unsigned char tmp_lab;
			train_label.read((char*)&tmp_lab, sizeof(unsigned char));

			std::vector<unsigned char> c(MNIST_IMAGE_SIZE);
			train_image.read((char*)&c[0], MNIST_IMAGE_SIZE * sizeof(unsigned char));
			if (train_image.fail())
			{
				printf("read error fail\n");
			}
			if (train_image.eof())
			{
				printf("read error eof\n");
			}

#pragma omp parallel for
			for (int j = 0; j < MNIST_IMAGE_SIZE; ++j) {
				tmp[3 * j + 0] = c[j];
				tmp[3 * j + 1] = c[j];
				tmp[3 * j + 2] = c[j];
			}
			char filename[256];
			sprintf(filename, "N%d_%03d.bmp", (int)tmp_lab, cnt);

			train.push_back(filename);

			train_buf.Add(tmp, tmp_lab, (std::string("train/") + filename));
			//stbi_write_bmp((dataDir + "/train/" + filename).c_str(), MNIST_IMAGE_WIDTH, MNIST_IMAGE_HIGHT, 3, (void*)tmp);
			cnt++;
		}
		//printf("train_buf %d\n", train_buf.buffers.size());
		train_buf.Flush();

		int addnum = 0;
		while (addnum != augment_num)
		{
			//printf("%d/%d            \r", addnum, augment_num);
			char filename[256];

			Augmentation aug(&mt, &d_rand);

			aug.gamma = 0.3;
			aug.rl = 0.0;
			aug.rnd_noize = 0.1;
			aug.rotation = 0.6;
			aug.rotation_max = 50.0;
			aug.sift = 0.6;
			aug.sift_max = 2;


			//ŒP—ûƒf[ƒ^‚Ì…‘‚µ
			for (int i = 0; i < train_num; i++)
			{
				if (addnum == augment_num) break;

				if (d_rand(mt) > 0.5) continue;
				const int tmp_lab = train_buf.buffers[i].label;

				const auto &tmp = train_buf.getBuffer(i);

				std::vector<std::vector<unsigned char>>& imageaug = ImageAugmentation(&tmp[0], MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, aug);

				img_greyscale grey;
				for (int i = 0; i < imageaug.size(); i++)
				{
					grey.greyscale(&imageaug[i][0], MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT);

					sprintf(filename, "N%d_%03d.bmp", (int)tmp_lab, cnt);
					train.push_back(filename);

					train_buf.Add(imageaug[i], tmp_lab, (std::string("augment/") + filename), true);
					//stbi_write_bmp((dataDir + "/train/" + filename).c_str(), MNIST_IMAGE_WIDTH, MNIST_IMAGE_HIGHT, 3, (void*)&imageaug[i][0]);
					cnt++;
					addnum++;
					if (addnum == augment_num) break;
				}
			}
		}
		train_label.close();
		train_image.close();

		train_buf.Flush(train_filelist);
	}

	if (!test_buf.disk_out_completed)
	{
		const int M = MNIST_IMAGE_TEST_NUM;
		ifstream test_image(dataDir + "\\t10k-images.idx3-ubyte", ios_base::binary);
		if (!test_image.is_open()) {
			cerr << "\"t10k-images.idx3-ubyte\" is not found!" << endl;
			return -1;
		}
		ifstream test_label(dataDir + "\\t10k-labels.idx1-ubyte", ios_base::binary);
		if (!test_label.is_open()) {
			cerr << "\"t10k-labels-idx1-ubyte\" is not found!" << endl;
			return -1;
		}
		test_image.seekg(4 * 4, ios_base::beg);
		test_label.seekg(4 * 2, ios_base::beg);
		for (int i = 0; i < M; ++i)
		{
			//if (i % 100 == 0) printf("test %d/%d                \r", i + 1, M);
			unsigned char tmp_lab;
			test_label.read((char*)&tmp_lab, sizeof(unsigned char));

			std::vector<unsigned char> c(MNIST_IMAGE_SIZE);
			test_image.read((char*)&c[0], MNIST_IMAGE_SIZE * sizeof(unsigned char));

#pragma omp parallel for
			for (int j = 0; j < MNIST_IMAGE_SIZE; ++j) {
				tmp[3 * j + 0] = c[j];
				tmp[3 * j + 1] = c[j];
				tmp[3 * j + 2] = c[j];
			}

			char filename[256];
			sprintf(filename, "N%d_%03d.bmp", (int)tmp_lab, i);

			test.push_back(filename);

			test_buf.Add(tmp, tmp_lab, (std::string("test/") + filename));
			//stbi_write_bmp((dataDir + "/test/" + filename).c_str(), MNIST_IMAGE_WIDTH, MNIST_IMAGE_HIGHT, 3, (void*)tmp);
		}
		test_buf.Flush(test_filelist);

		test_image.close();
		test_label.close();
	}

	train_buf.getDataList(train_filelist);
	test_buf.getDataList(test_filelist);
	return 0;
}

#endif
