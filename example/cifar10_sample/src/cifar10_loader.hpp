/*
Copyright (c) 2018, Sanaxn
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef _CIFAR10_LOADER_HPP

#define _CIFAR10_LOADER_HPP

#define CIFAR10_IMAGE_CHANNEL	3

#ifndef CIFAR10_IMAGE_WIDTH		//tiny-dnn --> cifar10_parser.h
#define CIFAR10_IMAGE_WIDTH		32	
#define CIFAR10_IMAGE_HEIGHT	32	
#define CIFAR10_IMAGE_SIZE	(CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HIGHT)
#endif

#define CIFAR10_IMAGE_TRAIN_NUM	50000	
#define CIFAR10_IMAGE_TRAIN_NUM_UNIT	10000	
#define CIFAR10_IMAGE_TEST_NUM	10000	

#include "image/ImageWorkBuffer.hpp"

inline void LoadCifar10_0(std::string dataDir, FILE* fp, int trainFlg, int& cnt, std::mt19937& mt, std::uniform_real_distribution<double>& d_rand, std::vector<std::string>& train, std::vector<std::string>& test, ImageWorkBufferList& train_buf, ImageWorkBufferList& test_buf);

inline int LoadCifar10(std::string dataDir, ImageWorkBufferList& train_buf, ImageWorkBufferList& test_buf, const size_t load_count, const int train_num, const int augment_num)
{
	std::mt19937 mt;
	std::uniform_real_distribution<double> d_rand;
	d_rand = std::uniform_real_distribution<double>(0.0, 1.0);

	std::vector<std::string> train;
	std::vector<std::string> test;

	std::string& train_filelist = std::string("train_list.txt");
	std::string& test_filelist = std::string("test_list.txt");

	train_buf.check_DiskOutCompleted(train_filelist);
	test_buf.check_DiskOutCompleted(test_filelist);

	std::vector<unsigned char> tmp(CIFAR10_IMAGE_SIZE * 3);

	int cnt = 0;
	if (!train_buf.disk_out_completed)
	{
		FILE* fp = fopen((dataDir + "/data_batch_1.bin").c_str(), "rb");

		for (int i = 0; i < CIFAR10_IMAGE_TRAIN_NUM_UNIT; i++)
		{
			LoadCifar10_0(dataDir, fp, 1, cnt, mt, d_rand, train, test, train_buf, test_buf);
		}
		fclose(fp);

		fp = fopen((dataDir + "/data_batch_2.bin").c_str(), "rb");
		for (int i = 0; i < CIFAR10_IMAGE_TRAIN_NUM_UNIT; i++)
		{
			LoadCifar10_0(dataDir, fp, 1, cnt, mt, d_rand, train, test, train_buf, test_buf);
		}
		fclose(fp);

		fp = fopen((dataDir + "/data_batch_3.bin").c_str(), "rb");
		for (int i = 0; i < CIFAR10_IMAGE_TRAIN_NUM_UNIT; i++)
		{
			LoadCifar10_0(dataDir, fp, 1, cnt, mt, d_rand, train, test, train_buf, test_buf);
		}
		fclose(fp);

		fp = fopen((dataDir + "/data_batch_4.bin").c_str(), "rb");
		for (int i = 0; i < CIFAR10_IMAGE_TRAIN_NUM_UNIT; i++)
		{
			LoadCifar10_0(dataDir, fp, 1, cnt, mt, d_rand, train, test, train_buf, test_buf);
		}
		fclose(fp);

		fp = fopen((dataDir + "/data_batch_5.bin").c_str(), "rb");
		for (int i = 0; i < CIFAR10_IMAGE_TRAIN_NUM_UNIT; i++)
		{
			LoadCifar10_0(dataDir, fp, 1, cnt, mt, d_rand, train, test, train_buf, test_buf);
		}
		fclose(fp);
		//printf("#train_buf %zd/%d\n", train_buf.buffers.size(), train_num);
		train_buf.Flush();


		int addnum = 0;
		while (addnum != augment_num)
		{
			char filename[256];
			const char id[][32] =
			{
				"airplane",

				"automobile",

				"bird",

				"cat",

				"deer",

				"dog",

				"frog",

				"horse",

				"ship",

				"truck",
			};

			Augmentation aug(&mt, &d_rand);

			aug.gamma = 0.4;
			aug.rl = 0.1;
			aug.color_nize = 0.1;
			aug.rotation = 0.2;
			aug.rotation_max = 45;
			aug.sift = 0.4;
			aug.sift_max = 3;
			aug.rnd_noize = 0.2;

			//ŒP—ûƒf[ƒ^‚Ì…‘‚µ
			for (int i = 0; i < train_num; i++)
			{
				if (addnum == augment_num) break;

				if (d_rand(mt) > 0.5) continue;
				int tmp_lab = train_buf.buffers[i].label;
				const auto &tmp = train_buf.getBuffer(i);

				std::vector<std::vector<unsigned char>>& imageaug = ImageAugmentation(&tmp[0], CIFAR10_IMAGE_WIDTH, CIFAR10_IMAGE_HEIGHT, aug);

				for (int i = 0; i < imageaug.size(); i++)
				{
					sprintf(filename, "%s_%03d.bmp", id[tmp_lab], cnt);
					train.push_back(filename);
					train_buf.Add(imageaug[i], tmp_lab, (std::string("augment/") + filename), true);
					cnt++;
					addnum++;
					if (addnum == augment_num) break;
				}
			}
			train_buf.Flush();
		}
		train_buf.Flush(train_filelist);
	}

	if (!test_buf.disk_out_completed) 
	{
		FILE* fp = fopen((dataDir + "/test_batch.bin").c_str(), "rb");
		for (int i = 0; i < CIFAR10_IMAGE_TEST_NUM; i++)
		{
			LoadCifar10_0(dataDir, fp, 0, cnt, mt, d_rand, train, test, train_buf, test_buf);
		}
		fclose(fp);
		test_buf.Flush();
		test_buf.Flush(test_filelist);
	}

	train_buf.getDataList(train_filelist);
	test_buf.getDataList(test_filelist);
	return 0;
}

inline void LoadCifar10_0(std::string dataDir, FILE* fp, int trainFlg, int& cnt, std::mt19937& mt, std::uniform_real_distribution<double>& d_rand, std::vector<std::string>& train, std::vector<std::string>& test, ImageWorkBufferList& train_buf, ImageWorkBufferList& test_buf)
{
	const char id[][32] =
	{
		"airplane",

		"automobile",

		"bird",

		"cat",

		"deer",

		"dog",

		"frog",

		"horse",

		"ship",

		"truck",
	};
	
	std::vector<unsigned char> buf((1 + CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT * CIFAR10_IMAGE_CHANNEL));
	size_t s = fread((void*)&buf[0], 1, (1 + CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT * CIFAR10_IMAGE_CHANNEL), fp);
	if (s != (1 + CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT * CIFAR10_IMAGE_CHANNEL))
	{
		fseek(fp, 0, SEEK_SET);
		//printf("%d ", s);
		s = fread((void*)&buf[0], 1, (1 + CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT * CIFAR10_IMAGE_CHANNEL), fp);
		//printf("read %d ->%d\n", (1 + CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HIGHT * CIFAR10_IMAGE_CHANNEL), s);
	}

	const char label = buf[0];
	//printf("%s\n", id[label]);

	const int x = CIFAR10_IMAGE_WIDTH, y = CIFAR10_IMAGE_HEIGHT;
	const int nbit = CIFAR10_IMAGE_CHANNEL;

	const unsigned char* image = ((unsigned char*)&buf[1]);
	const unsigned char* R = image;
	const unsigned char* G = R + CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT;
	const unsigned char* B = G + CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT;

	std::vector<unsigned char> data(3 * x * y);
#pragma omp parallel for
	for (int i = 0; i < x*y; i++) {
		data[i * 3 + 0] = R[i];
		data[i * 3 + 1] = G[i];
		data[i * 3 + 2] = B[i];
	}

	char filename[256];
	if (trainFlg)
	{
		sprintf(filename, "%s_%03d.bmp", id[label], cnt);
		train.push_back(filename);
		train_buf.Add(data, label, (std::string("train/") + filename));
	}
	else
	{
		sprintf(filename, "%s_%03d.bmp", id[label], cnt);
		test.push_back(filename);
		test_buf.Add(data, label, (std::string("test/") + filename));
	}
	//stbi_write_bmp(filename, CIFAR10_IMAGE_WIDTH, CIFAR10_IMAGE_HIGHT, 3, (void*)&data[0]);
	cnt++;
}


#endif
