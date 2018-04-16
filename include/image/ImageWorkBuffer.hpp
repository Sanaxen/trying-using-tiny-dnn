/*
Copyright (c) 2018, Sanaxn
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef _IMAGEWORKBUFFER_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <random>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#endif

inline float_t* whitening(Image* img)
{
	float_t* whitening_img = image_whitening<float_t>(img);
	return whitening_img;

}

class ImageWorkBuffer
{
public:
	std::vector<unsigned char> buffer;
	int label;
	std::string filenames;
	bool augment_data = false;

	inline ImageWorkBuffer() {}
	inline ImageWorkBuffer(std::vector<unsigned char>& buffer_, int label_, std::string& filenames_, bool augment_= false) :
		buffer(buffer_), label(label_), filenames(filenames_), augment_data(augment_)
	{}

	inline std::vector<unsigned char> getBuffer(const std::string& data_path)
	{
		if (buffer.size() > 0)
		{
			return buffer;
		}
		Image* img = readImage((data_path + "/" + filenames).c_str());

		std::vector<unsigned char>buf(3 * img->height*img->width);

		for (int i = 0; i < img->height; i++)
		{
			for (int j = 0; j < img->width; j++)
			{
				int pos = (i*img->width + j);

				buf[3 * pos + 0] = img->data[pos].r;
				buf[3 * pos + 1] = img->data[pos].g;
				buf[3 * pos + 2] = img->data[pos].b;
			}
		}
		delete img;
		return buf;
	}

};

class ImageWorkBufferList
{
public:
	bool disk_out_completed = false;
	std::string data_path;
	int size_x;
	int size_y;
	std::vector<ImageWorkBuffer> buffers;

	std::mt19937 mt;
	size_t data_num = 0;

	inline ~ImageWorkBufferList()
	{
		clear(true);
	}

	inline void check_DiskOutCompleted(std::string& filelist)
	{
		data_num = 0;
		disk_out_completed = false;
		FILE* fp = fopen((data_path + "/" + filelist).c_str(), "r");
		if (fp)
		{
			disk_out_completed = true;
			printf("image disk_out_completed!!\n");
			fclose(fp);
		}
	}

	inline ImageWorkBufferList(int x, int y, std::string& data_path_)
	{
		size_x = x;
		size_y = y;
		data_path = data_path_;
		mt = std::mt19937(13);
	}
	inline void Add(std::vector<unsigned char>& img, int label, std::string& filename, bool augment_data = false)
	{
		buffers.push_back(ImageWorkBuffer(img, label, filename, augment_data));
		if (!disk_out_completed && buffers.size() % 10000 == 0)
		{
			Flush();
		}

	}

	inline void Flush()
	{
		printf("...output image...");
		const size_t sz = buffers.size();
#pragma omp parallel for
		for (int i = 0; i < sz; ++i)
		{
			if (buffers[i].buffer.size() == 0) continue;
			const std::vector<unsigned char>& tmp = buffers[i].buffer;
			int stat = stbi_write_bmp((data_path + "/" + buffers[i].filenames).c_str(), size_x, size_y, 3, &tmp[0]);

			buffers[i].buffer.resize(0);
			buffers[i].buffer.shrink_to_fit();
		}
		printf("end\n");
	}

	inline void Flush(std::string& filelist)
	{
		if (!disk_out_completed)
		{
			std::random_device rdev{};
			std::mt19937 mt(rdev());
			std::shuffle(buffers.begin(), buffers.end(), mt);

			FILE* fp = fopen((data_path + "/" + filelist).c_str(), "w");
			for (int i = 0; i < buffers.size(); i++)
			{
				fprintf(fp, "%d >%d >%s\n", buffers[i].label, buffers[i].augment_data?1:0, buffers[i].filenames.c_str());
			}
			fclose(fp);
			disk_out_completed = true;
		}
	}

	inline bool getDataList(std::string& filelist)
	{
		data_num = 0;
		if (!disk_out_completed) return false;

		char buf[640];
		FILE* fp = fopen((data_path + "/" + filelist).c_str(), "r");
		int cnt = 0;
		while (fgets(buf, 640, fp) != NULL)
		{
			int augment = 0;
			int tmp_lab = -1;
			sscanf(buf, "%d >", &tmp_lab);
			char* beg = strchr(buf, '>');
			beg++;

			sscanf(beg, "%d >", &augment);
			beg = strchr(beg, '>');
			beg++;

			char* end = strchr(beg, '\n');
			*end = '\0';
			std::vector<unsigned char> tmp(0);
			Add(tmp, tmp_lab, std::string(beg), augment ? 1 : 0);
			data_num++;
			//printf("[%d]%s                                  \r", ++cnt, beg);
		}
		//printf("\n");
		fclose(fp);

		return true;
	}

	inline std::vector<unsigned char> getBuffer(const int index)
	{
		return buffers[index].getBuffer(data_path);
	}

	inline Image* load(const int index)
	{
		return readImage((data_path + "/" + buffers[index].filenames).c_str());
	}

	inline void image_dump(int isForce = false)
	{
		if (!isForce)
		{
			FILE* fp = fopen((data_path + "/" + buffers[0].filenames).c_str(), "r");
			if (fp)
			{
				fclose(fp);
				return;
			}
		}

		printf("...output image...");
#pragma omp parallel for
		for (int i = 0; i < buffers.size(); ++i)
		{
			std::vector<unsigned char>& tmp = buffers[i].buffer;
			if (tmp.size() == 0)
			{
				Image* img = load(i);
				ImageWrite((data_path + "/" + buffers[i].filenames).c_str(), img);
				delete img;
			}
			else
			{
				int stat = stbi_write_bmp((data_path + "/" + buffers[i].filenames).c_str(), size_x, size_y, 3, &tmp[0]);
			}
			//printf("%d[%s]\n", stat, filenames[i].c_str());
		}
		printf("end\n");
	}

	inline void clear(bool Force = false)
	{
		if (!Force) return;
#pragma omp parallel for
		for (int i = 0; i < buffers.size(); ++i)
		{
			buffers[i].buffer.resize(0);
			buffers[i].buffer.shrink_to_fit();
		}
		buffers.resize(0);
		buffers.shrink_to_fit();
	}
};

template<class T>
inline void image_to_vec(const T* img, const int w, const int h, const int channel, tiny_dnn::vec_t& vec, const float_t scale_min = -1.0, const float_t scale_max = -1.0)
{
	bool use_scale = (scale_min < scale_max);

	for (int c = 0; c < channel; c++)
	{
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				const int ii = y*w + x;
				if (use_scale)
				{
					vec[c * h*w + y*w + x] = scale_min + (scale_max - scale_min) *img[3 * ii + c] / 255.0;
				}
				else
				{
					vec[c * h*w + y*w + x] = img[3 * ii + c];
				}
			}
		}
	}
}


inline void set_image_data(ImageWorkBufferList& buf, std::vector<tiny_dnn::label_t>& labels, std::vector<tiny_dnn::vec_t>& images, const int seed, const int channel, const int padding)
{
	const int M = images.size();

	size_t data_size_max = buf.data_num;
	std::uniform_int_distribution<int> rand(0, data_size_max - 1);

#pragma omp parallel for
	for (int k = 0; k < M; ++k)
	{
		size_t i = k;


		std::vector<unsigned char> imgbuf = buf.buffers[i].buffer;
		if (imgbuf.size() == 0)
		{
			i = rand(buf.mt);
			imgbuf = buf.getBuffer(i);
		}
		//fprintf(stderr, "@%d/%d           \n", (i + 1), M);
		Image* img = ToImage(&imgbuf[0], buf.size_x, buf.size_y);
		img_padding pad;
#if 10
		pad.padding(img, padding, 0);

		tiny_dnn::vec_t tmp(img->height*img->width*channel);
		tiny_dnn::label_t tmp_lb = buf.buffers[i].label;

		float_t* whitening_img = whitening(img);
		image_to_vec(whitening_img, img->width, img->height, channel, tmp);
#else
		float_t scale_min = -1.0;
		float_t scale_max = 1.0;

		pad.padding(img, padding, 0);

		tiny_dnn::vec_t tmp(img->height*img->width*channel);
		tiny_dnn::label_t tmp_lb = buf.buffers[i].label;

		float_t* whitening_img = ImageTo<float_t>(img);
		image_to_vec(whitening_img, img->width, img->height, channel, tmp, scale_min, scale_max);
#endif
		delete[] whitening_img;

		images[k].resize(0);
		images[k].shrink_to_fit();

		images[k] = tmp;
		labels[k] = tmp_lb;

		delete img;
	}
}

#endif
