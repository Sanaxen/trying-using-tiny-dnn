/*
Copyright (c) 2018, Sanaxn
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef __IMAGE_HPP

#undef __IMAGE_HPP

#include "image/gcn_normalizedData.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb/stb_image_write.h"
#endif

#pragma warning(disable : 4244)
#pragma warning(disable : 4018)

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

using namespace std;

typedef struct Rgb_ {
	double b;
	double g;
	double r;
	double alp;
	~Rgb_() {}
	Rgb_() {}
	inline Rgb_(int x, int y, int z)
	{
		r = x;
		g = y;
		b = z;
		alp = 255;
	}
	inline Rgb_(const int* x)
	{
		r = x[0];
		g = x[1];
		b = x[2];
		alp = 255;
	}
	inline Rgb_(const unsigned char* x)
	{
		r = x[0];
		g = x[1];
		b = x[2];
		alp = 255;
	}
	inline Rgb_(const unsigned char x)
	{
		r = x;
		g = x;
		b = x;
		alp = 255;
	}
} Rgb;


class Image
{
public:
	unsigned int height;
	unsigned int width;
	Rgb *data;

	Image()
	{
		data = 0;
	}
	~Image()
	{
		if (data) delete[] data;
	}
};

inline void ImageWrite(const char* filename, Image* img)
{
	std::vector<unsigned char> data(3 * img->height*img->width);

	for (int i = 0; i < img->height; i++)
	{
		for (int j = 0; j < img->width; j++)
		{
			int pos = (i*img->width + j);

			data[3 * pos + 0] = img->data[pos].r;
			data[3 * pos + 1] = img->data[pos].g;
			data[3 * pos + 2] = img->data[pos].b;
		}
	}
	stbi_write_bmp(filename, img->width, img->height, 3, (void*)&data[0]);
}

template<class T>
inline Image* ToImage(T* data, int x, int y)
{
	Image *img = 0;

	img = new Image;
	img->data = new Rgb[x*y];
	memset(img->data, '\0', sizeof(Rgb)*x*y);
	img->height = y;
	img->width = x;

#pragma omp parallel for
	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			int pos = i*x + j;

			img->data[pos].r = data[3 * pos + 0];
			img->data[pos].g = data[3 * pos + 1];
			img->data[pos].b = data[3 * pos + 2];
		}
	}
	return img;
}

template<class T>
inline T* ImageTo(Image* img)
{
	T* data = new T[img->width*img->height * 3];

#pragma omp parallel for
	for (int i = 0; i < img->height; i++)
	{
		for (int j = 0; j < img->width; j++)
		{
			int pos = (i*img->width + j);

			data[3 * pos + 0] = img->data[pos].r;
			data[3 * pos + 1] = img->data[pos].g;
			data[3 * pos + 2] = img->data[pos].b;
		}
	}
	return data;
}

inline Image* readImage(const char *filename)
{
	//int i, j;
	//int real_width;
	//unsigned int width, height;
	//unsigned int color;
	//FILE *fp;
	//unsigned char *bmp_line_data;
	Image *img;

	unsigned char *data = 0;
	int x, y;
	int nbit;
	data = stbi_load(filename, &x, &y, &nbit, 0);
	if (data == NULL)
	{
		printf("image file[%s] read error.\n", filename);
		return NULL;
	}
	//printf("height %d   width %d \n", y, x);

	img = new Image;
	img->data = new Rgb[x*y];
	memset(img->data, '\0', sizeof(Rgb)*x*y);
	img->height = y;
	img->width = x;

#pragma omp parallel for
	for (int i = 0; i<y; ++i) {
		for (int j = 0; j<x; ++j) {
			if (nbit == 1)	//8bit
			{
				int pos = (i*x + j);
				img->data[pos].r = data[pos];
				img->data[pos].g = data[pos];
				img->data[pos].b = data[pos];
				img->data[pos].alp = 255;
			}
			if (nbit == 2)	//16bit
			{
				int pos = (i*x + j);
				img->data[pos].r = data[pos * 2 + 0];
				img->data[pos].g = data[pos * 2 + 0];
				img->data[pos].b = data[pos * 2 + 0];
				img->data[pos].alp = data[pos * 2 + 1];
			}
			if (nbit == 3)	//24
			{
				int pos = (i*x + j);
				img->data[pos].r = data[pos * 3 + 0];
				img->data[pos].g = data[pos * 3 + 1];
				img->data[pos].b = data[pos * 3 + 2];
				img->data[pos].alp = 255;
			}
			if (nbit == 4)	//32
			{
				int pos = (i*x + j);
				img->data[pos].r = data[pos * 4 + 0];
				img->data[pos].g = data[pos * 4 + 1];
				img->data[pos].b = data[pos * 4 + 2];
				img->data[pos].alp = data[pos * 4 + 3];
			}
		}
	}
	stbi_image_free(data);

	return img;
}

//PCA on Normalized Data
template<class T>
inline T* image_whitening(Image* img, double eps = 0.0)
{
	double av = 0.0;
	const int sz = img->height*img->width;
	T* data = ImageTo<T>(img);

	T* ret =  gcn_normalizedData(data, sz * 3, eps);
	delete[] data;

	return ret;
//
//#pragma omp parallel for reduction(+:av)
//	for (int k = 0; k < sz; k++)
//	{
//		data[3*k+0] = img->data[k].r / 255.0;
//		data[3*k+1] = img->data[k].g / 255.0;
//		data[3*k+2] = img->data[k].b / 255.0;
//		av += data[3*k+0];
//		av += data[3*k+1];
//		av += data[3*k+2];
//	}
//	av /= (double)(3 * sz);
//
//	double sd = 0.0;
//#pragma omp parallel for reduction(+:sd)
//	for (int k = 0; k < sz; k++)
//	{
//		sd += pow(data[3*k+0] - av, 2.0);
//		sd += pow(data[3*k+1] - av, 2.0);
//		sd += pow(data[3*k+2] - av, 2.0);
//	}
//	sd = sqrt(sd / (double)(3 * sz)) + eps;
//	//if ( fabs(sd) < 1.0e-16 ) return;
//
//#pragma omp parallel for
//	for (int k = 0; k < sz; k++)
//	{
//		data[3*k+0] = (data[3*k+0] - av) / sd;
//		data[3*k+1] = (data[3*k+1] - av) / sd;
//		data[3*k+2] = (data[3*k+2] - av) / sd;
//	}
//
//	return data;
}

class img_greyscale
{
public:
	void greyscale(Image* img)
	{
#pragma omp parallel for
		for (int i = 0; i < img->height*img->width; i++)
		{
			double c = (0.299 * img->data[i].r + 0.587 * img->data[i].g + 0.114 * img->data[i].b);
			img->data[i].r = c;
			img->data[i].g = c;
			img->data[i].b = c;
		}
	}
	void greyscale(double* data, int x, int y)
	{
#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			double c = (0.299 * data[3 * i + 0] + 0.587 * data[3 * i + 1] + 0.114 * data[3 * i + 2]);
			data[3 * i + 0] = c;
			data[3 * i + 1] = c;
			data[3 * i + 2] = c;
		}
	}
	void greyscale(unsigned char* data, int x, int y)
	{
#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			double c = (0.299 * data[3 * i + 0] + 0.587 * data[3 * i + 1] + 0.114 * data[3 * i + 2]);
			data[3 * i + 0] = c;
			data[3 * i + 1] = c;
			data[3 * i + 2] = c;
		}
	}

};
class img_gamma
{
	double gamma_;
public:
	img_gamma(double gamma)
	{
		gamma_ = gamma;
	}
	void gamma(Image* img)
	{
#pragma omp parallel for
		for (int i = 0; i < img->height*img->width; i++)
		{
			img->data[i].r = 255 * pow(img->data[i].r / 255.0, 1.0 / gamma_);
			img->data[i].g = 255 * pow(img->data[i].g / 255.0, 1.0 / gamma_);
			img->data[i].b = 255 * pow(img->data[i].b / 255.0, 1.0 / gamma_);
		}
	}
	void gamma(double* data, int x, int y)
	{
#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			data[3 * i + 0] = 255 * pow(data[3 * i + 0] / 255.0, 1.0 / gamma_);
			data[3 * i + 1] = 255 * pow(data[3 * i + 1] / 255.0, 1.0 / gamma_);
			data[3 * i + 2] = 255 * pow(data[3 * i + 2] / 255.0, 1.0 / gamma_);
		}
	}
	void gamma(unsigned char* data, int x, int y)
	{
#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			data[3 * i + 0] = 255 * pow(data[3 * i + 0] / 255.0, 1.0 / gamma_);
			data[3 * i + 1] = 255 * pow(data[3 * i + 1] / 255.0, 1.0 / gamma_);
			data[3 * i + 2] = 255 * pow(data[3 * i + 2] / 255.0, 1.0 / gamma_);
		}
	}
};

class img_contrast
{
	int min_table;
	int max_table;
	int diff_table;

	float LUT_HC[255];
	float LUT_LC[255];

public:
	img_contrast()
	{
		// ルックアップテーブルの生成
		min_table = 50;
		max_table = 205;
		diff_table = max_table - min_table;

		//ハイコントラストLUT作成
		for (int i = 0; i < min_table; i++)	LUT_HC[i] = 0;
		for (int i = min_table; i < max_table; i++)	LUT_HC[i] = 255.0 * (i - min_table) / (float)diff_table;
		for (int i = max_table; i < 255; i++)	LUT_HC[i] = 255.0;

		// ローコントラストLUT作成
		for (int i = 0; i < 255; i++) LUT_LC[i] = min_table + i * (diff_table) / 255.0;
	}

	void high(Image* img)
	{
#pragma omp parallel for
		for (int i = 0; i < img->height*img->width; i++)
		{
			img->data[i].r = (unsigned char)(std::min(255.0, LUT_HC[(unsigned char)img->data[i].r] * img->data->r));
			img->data[i].g = (unsigned char)(std::min(255.0, LUT_HC[(unsigned char)img->data[i].g] * img->data->r));
			img->data[i].b = (unsigned char)(std::min(255.0, LUT_HC[(unsigned char)img->data[i].b] * img->data->r));
		}
	}
	void high(double* data, int x, int y)
	{
#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			data[3 * i + 0] = (unsigned char)(std::min(255.0, LUT_HC[(int)std::max(0.0, std::max(255.0, data[3 * i + 0]))] * data[3 * i + 0]));
			data[3 * i + 1] = (unsigned char)(std::min(255.0, LUT_HC[(int)std::max(0.0, std::max(255.0, data[3 * i + 1]))] * data[3 * i + 1]));
			data[3 * i + 2] = (unsigned char)(std::min(255.0, LUT_HC[(int)std::max(0.0, std::max(255.0, data[3 * i + 2]))] * data[3 * i + 2]));
		}
	}
	void high(unsigned char* data, int x, int y)
	{
#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			data[3 * i + 0] = LUT_HC[data[3 * i + 0]] * data[3 * i + 0];
			data[3 * i + 1] = LUT_HC[data[3 * i + 1]] * data[3 * i + 1];
			data[3 * i + 2] = LUT_HC[data[3 * i + 2]] * data[3 * i + 2];
		}
	}
	void low(Image* img)
	{
#pragma omp parallel for
		for (int i = 0; i < img->height*img->width; i++)
		{
			img->data[i].r = (unsigned char)(std::min(255.0, LUT_LC[(unsigned char)img->data[i].r] * img->data->r));
			img->data[i].g = (unsigned char)(std::min(255.0, LUT_LC[(unsigned char)img->data[i].g] * img->data->g));
			img->data[i].b = (unsigned char)(std::min(255.0, LUT_LC[(unsigned char)img->data[i].b] * img->data->b));
		}
	}
	void low(double* data, int x, int y)
	{
#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			data[3 * i + 0] = (unsigned char)(std::min(255.0, LUT_LC[(int)std::max(0.0, std::min(255.0, data[3 * i + 0]))] * data[3 * i + 0]));
			data[3 * i + 1] = (unsigned char)(std::min(255.0, LUT_LC[(int)std::max(0.0, std::min(255.0, data[3 * i + 1]))] * data[3 * i + 1]));
			data[3 * i + 2] = (unsigned char)(std::min(255.0, LUT_LC[(int)std::max(0.0, std::min(255.0, data[3 * i + 2]))] * data[3 * i + 2]));
		}
	}
	void low(unsigned char* data, int x, int y)
	{
#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			data[3 * i + 0] = LUT_LC[data[3 * i + 0]] * data[3 * i + 0];
			data[3 * i + 1] = LUT_LC[data[3 * i + 1]] * data[3 * i + 1];
			data[3 * i + 2] = LUT_LC[data[3 * i + 2]] * data[3 * i + 2];
		}
	}
};

class img_noize
{
	std::mt19937 mt;
	double sigma_;
	std::uniform_real_distribution<double> rand_a;
	double r;
public:
	img_noize(double sigma = 15.0, double r_ = 0.3)
	{
		std::random_device seed_gen;
		std::mt19937 engine(seed_gen());
		mt = engine;
		sigma_ = sigma;
		std::uniform_real_distribution<double> rand_aa(0.0, 1.0);
		rand_a = rand_aa;
		r = r_;
	}

	void noize(Image* img)
	{
		std::uniform_real_distribution<double> d_rand(-sigma_, sigma_);

#pragma omp parallel for
		for (int i = 0; i < img->height*img->width; i++)
		{
			if (rand_a(mt) < r)
			{
				img->data[i].r = (unsigned char)(std::max(0.0, std::min(255.0, img->data[i].r + d_rand(mt))));
				img->data[i].g = (unsigned char)(std::max(0.0, std::min(255.0, img->data[i].g + d_rand(mt))));
				img->data[i].b = (unsigned char)(std::max(0.0, std::min(255.0, img->data[i].b + d_rand(mt))));
			}
		}
	}
	void noize(double* data, int x, int y)
	{
		std::uniform_real_distribution<double> d_rand(-sigma_, sigma_);

#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			if (rand_a(mt) < r)
			{
				data[3 * i + 0] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 0] + d_rand(mt))));
				data[3 * i + 1] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 1] + d_rand(mt))));
				data[3 * i + 2] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 2] + d_rand(mt))));
			}
		}
	}
	void noize(unsigned char* data, int x, int y)
	{
		std::normal_distribution<double> d_rand(0.0, sigma_);

#pragma omp parallel for
		for (int i = 0; i < x*y; i++)
		{
			if (rand_a(mt) < r)
			{
				data[3 * i + 0] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 0] + d_rand(mt))));
				data[3 * i + 1] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 1] + d_rand(mt))));
				data[3 * i + 2] = (unsigned char)(std::max(0.0, std::min(255.0, data[3 * i + 2] + d_rand(mt))));
			}
		}
	}
};

class img_filter
{
	double weight[3][3];
public:
	img_filter(double* filter = NULL)
	{
		if (filter)
		{
			weight[0][0] = filter[0];
			weight[0][1] = filter[1];
			weight[0][2] = filter[2];
			weight[1][0] = filter[3];
			weight[1][1] = filter[4];
			weight[1][2] = filter[5];
			weight[2][0] = filter[6];
			weight[2][1] = filter[7];
			weight[2][2] = filter[8];
		}
		else
		{
			weight[0][0] = 1.0;
			weight[0][1] = 1.0;
			weight[0][2] = 1.0;
			weight[1][0] = 1.0;
			weight[1][1] = 1.0;
			weight[1][2] = 1.0;
			weight[2][0] = 1.0;
			weight[2][1] = 1.0;
			weight[2][2] = 1.0;
		}
	}

	void filter(Image* img)
	{
		const int x = img->width;
		const int y = img->height;

		for ( int i = 0; i < y; i++ )
		{
			for ( int j = 0; j < x; j++ )
			{
				double r, g, b;
				
				r = g = b = 0.0;
				for ( int ii = 0; ii < 3; ii++ )
				{
					for ( int jj = 0; jj < 3; jj++ )
					{
							int pos = ((i+ii)*x + (j+jj));
							if (pos >= x*y) continue;
							r += img[pos].data->r * weight[ii][jj];
							g += img[pos].data->g * weight[ii][jj];
							b += img[pos].data->b * weight[ii][jj];
					}
				}
				r /= 9.0;
				g /= 9.0;
				b /= 9.0;
				int pos = i*x + j;
				img[pos].data->r = (unsigned char)std::min(255.0, r);
				img[pos].data->g = (unsigned char)std::min(255.0, g);
				img[pos].data->b = (unsigned char)std::min(255.0, b);
			}
		}
	}
	void filter(double* data, int x, int y)
	{
		Image* img = ToImage(data, x, y);
		filter(img);

		double* data2 = ImageTo<double>(img);
#pragma omp parallel for
		for (int i = 0; i < x*y * 3; i++)
		{
			data[i] = data2[i];
		}
		delete[] data2;
		delete img;
	}
	void filter(unsigned char* data, int x, int y)
	{
		Image* img = ToImage(data, x, y);
		filter(img);

		unsigned char* data2 = ImageTo<unsigned char>(img);
#pragma omp parallel for
		for (int i = 0; i < x*y * 3; i++)
		{
			data[i] = data2[i];
		}
		delete[] data2;
		delete img;
	}
};


class img_rotation
{
public:
	img_rotation() {}

	void rotation(Image* img, const double rad)
	{
		const int x = img->width;
		const int y = img->height;
		double centerX = x / 2.0;
		double centerY = y / 2.0;
		double cosRadian = cos(rad);
		double sinRadian = sin(rad);

		//対角の長さに画像を拡大しておく
		double r = sqrt(centerX*centerX + centerY*centerY)*2.0;
		int R = (int)(r + 0.5);

		//増量分
		int fx = (R - x) / 2;
		int fy = (R - y) / 2;

		int Rx = x + 2 * fx;
		int Ry = y + 2 * fy;
		Image* RimgI = new Image;
		RimgI->data = new Rgb[Rx*Ry];
		RimgI->height = Ry;
		RimgI->width = Rx;

		//拡大領域に複写
#pragma omp parallel for
		for (int i = 0; i < Ry; i++)
		{
			for (int j = 0; j < Rx; j++)
			{
				int pos = i*Rx + j;
				if (i - fy >= 0 && i - fy < y && j - fx >= 0 && j - fx < x)
				{
					RimgI->data[pos] = img->data[(i - fy) * x + (j - fx)];
				}
				else
				{
					//拡大して広げたところは境界値で埋める
					int ii = -1, jj = -1;

					if (i - fy < 0)  ii = 0;
					if (i - fy >= y) ii = y - 1;

					if (j - fx < 0)  jj = 0;
					if (j - fx >= x) jj = x - 1;

					if (ii < 0) ii = 0;
					if (jj < 0) jj = 0;

					pos = i*Rx + j;
					RimgI->data[pos] = img->data[ii * x + jj];
				}
			}
		}

		//回転させるための複製
		Rgb* data = new Rgb[Rx*Ry];
		memcpy(data, RimgI->data, sizeof(Rgb)*Rx*Ry);

		centerX = Rx / 2.0;
		centerY = Ry / 2.0;
#pragma omp parallel for
		for (int i = 0; i < Ry; i++)
		{
			for (int j = 0; j < Rx; j++)
			{
				int pos = i*Rx + j;
				
				int pointX = (int)((j - centerX) * cosRadian - (i - centerY) * sinRadian + centerX);
				int pointY = (int)((j - centerX) * sinRadian + (i - centerY) * cosRadian + centerY);

				// poiuntX, pointYが入力画像の有効範囲にあれば出力画像へ代入する
				if (pointX >= 0 && pointX < Rx && pointY >= 0 && pointY < Ry) {
					RimgI->data[pos] = data[pointY * Rx + pointX];
				}
				else {
					RimgI->data[pos] = Rgb(0,0,0);
				}
			}
		}

		//元のサイズで切り出し
#pragma omp parallel for
		for (int i = 0; i < Ry; i++)
		{
			for (int j = 0; j < Rx; j++)
			{
				int pos = i*Rx + j;
				if (i - fy >= 0 && i - fy < y && j - fx >= 0 && j - fx < x)
				{
					img->data[(i - fy) * x + (j - fx)] = RimgI->data[pos];
				}
			}
		}
		delete[] data;
		delete RimgI;
	}

	void rotation(double* data, int x, int y, const double rad)
	{
		Image* img = ToImage(data, x, y);
		rotation(img, rad);

		double* data2 = ImageTo<double>(img);
#pragma omp parallel for
		for (int i = 0; i < x*y*3; i++)
		{
			data[i] = data2[i];
		}
		delete img;
		delete[] data2;
	}

	void rotation(unsigned char* data, int x, int y, const double rad)
	{
		Image* img = ToImage(data, x, y);
		rotation(img, rad);

		unsigned char* data2 = ImageTo<unsigned char>(img);
#pragma omp parallel for
		for (int i = 0; i < x*y * 3; i++)
		{
			data[i] = data2[i];
		}
		delete img;
		delete[] data2;
	}
};
class img_sift
{
public:
	img_sift() {}

	void sift(Image* img, const int axis, const int delta)
	{
		const int x = img->width;
		const int y = img->height;

		Rgb* data = new Rgb[x*y];
		memcpy(data, img->data, sizeof(Rgb)*x*y);

		if (axis == 1)
		{
			for (int i = 0; i < y; i++)
			{
				for (int j = 0; j < x - delta; j++)
				{
					int pos = i*x + j;

					img->data[pos] = data[i * x + (j + delta)];
				}
				for (int j = x- delta; j < x; j++)
				{
					int pos = i*x + j;

					img->data[pos] = data[i * x + (x-1)];
				}
			}
		}
		if (axis == -1)
		{
			for (int i = 0; i < y; i++)
			{
				for (int j = 0; j < x; j++)
				{
					int pos = i*x + j + delta;

					if (j + delta == x) break;
					img->data[pos] = data[i * x + j];
				}
				for (int j = 0; j < delta; j++)
				{
					int pos = i*x + j;

					img->data[pos] = data[i * x + 0];
				}
			}
		}
		if (axis == 2)
		{
			for (int i = 0; i < y - delta; i++)
			{
				for (int j = 0; j < x; j++)
				{
					int pos = i*x + j;

					img->data[pos] = data[(i+delta) * x + j];
				}
			}
			for (int i = y - delta; i < y; i++)
			{
				for (int j = 0; j < x; j++)
				{
					int pos = i*x + j;

					img->data[pos] = data[(y-1) * x + j];
				}
			}
		}
		if (axis == -2)
		{
			for (int i = delta; i < y; i++)
			{
				for (int j = 0; j < x; j++)
				{
					int pos = i *x + j;

					img->data[pos] = data[ (i-delta)* x + j];
				}
			}
			for (int i = 0; i < delta; i++)
			{
				for (int j = 0; j < x; j++)
				{
					int pos = i*x + j;

					img->data[pos] = data[0 * x + j];
				}
			}
		}
		delete[] data;
	}

	void sift(double* data, int x, int y, const int axis, const int delta)
	{
		Image* img = ToImage(data, x, y);
		sift(img, axis, delta);

		double* data2 = ImageTo<double>(img);
#pragma omp parallel for
		for (int i = 0; i < x*y * 3; i++)
		{
			data[i] = data2[i];
		}
		delete[] data2;
		delete img;
	}

	void sift(unsigned char* data, int x, int y, const int axis, const int delta)
	{
		Image* img = ToImage(data, x, y);
		sift(img, axis, delta);

		unsigned char* data2 = ImageTo<unsigned char>(img);
#pragma omp parallel for
		for (int i = 0; i < x*y * 3; i++)
		{
			data[i] = data2[i];
		}
		delete[] data2;
		delete img;
	}
};


class img_padding
{
public:
	img_padding() {}

	void padding(Image* img, const int padding_sz, const float padding_value)
	{
		const int x = img->width;
		const int y = img->height;
		int Rx = x + 2 * padding_sz;
		int Ry = y + 2 * padding_sz;

		Image* RimgI = new Image;
		RimgI->data = new Rgb[Rx*Ry];
		RimgI->height = Ry;
		RimgI->width = Rx;

#pragma omp parallel for
		for (int i = 0; i < Ry; i++)
		{
			for (int j = 0; j < Rx; j++)
			{
				int pos = i*Rx + j;
				if (i - padding_sz >= 0 && i - padding_sz < y && j - padding_sz >= 0 && j - padding_sz < x)
				{
					RimgI->data[pos] = img->data[(i - padding_sz) * x + (j - padding_sz)];
				}
				else
				{
					//拡大して広げたところはpadding_value
					pos = i*Rx + j;
					RimgI->data[pos] = Rgb(padding_value, padding_value, padding_value);
				}
			}
		}
		delete[] img->data;
		img->data = RimgI->data;
		img->height = RimgI->height;
		img->width = RimgI->width;
	}

	void padding(double** data, int x, int y, const int padding_sz, const float padding_value)
	{
		Image* img = ToImage<double>(*data, x, y);
		padding(img, padding_sz, padding_value);

		delete[] * data;
		*data = ImageTo<double>(img);
	}

	void padding(unsigned char** data, int x, int y, const int padding_sz, const float padding_value)
	{
		Image* img = ToImage<unsigned char>(*data, x, y);
		padding(img, padding_sz, padding_value);

		delete[] * data;
		*data = ImageTo<unsigned char>(img);
	}
};

class Augmentation
{
public:
	std::mt19937* mt;
	std::uniform_real_distribution<double>* d_rand;

	Augmentation()
	{

	}
	Augmentation(std::mt19937* mt_, std::uniform_real_distribution<double>* d_rand_)
	{
		mt = mt_;
		d_rand = d_rand_;

		gamma = 0.0;
		rl = 0.0;
		color_nize = 0.0;
		rnd_noize = 0.0;
		rotation = 0.0;
		rotation_max = 10.0;
		sift = 0.0;
		sift_max = 4.0;
	}
	double gamma;
	double rl;
	double color_nize;
	double rnd_noize;
	double rotation;
	double rotation_max;
	double sift;
	double sift_max;

	inline double rnd()
	{
		double g = (*d_rand)(*mt);
		return g;
	}
};

std::vector<std::vector<unsigned char>> ImageAugmentation(const unsigned char* data, const int x, const int y, Augmentation& aug)
{
	//訓練データの水増し
	std::vector<std::vector<unsigned char>>image_augmentat;

	if (aug.gamma > 0.0)
	{
		std::vector<unsigned char>data2(3 * x * y, 0);

		double g;
		if ((g = aug.rnd()) < aug.gamma)
		{
			g = 1.2 - g*2.0;
			//ガンマ補正
#pragma omp parallel for
			for (int i = 0; i < x*y; i++) {
				data2[i * 3 + 0] = 255 * pow(data[i * 3 + 0] / 255.0, 1.0 / g);
				data2[i * 3 + 1] = 255 * pow(data[i * 3 + 1] / 255.0, 1.0 / g);
				data2[i * 3 + 2] = 255 * pow(data[i * 3 + 2] / 255.0, 1.0 / g);
			}
			image_augmentat.push_back(data2);
		}
	}

	if (aug.rl > 0.0 && aug.rnd() < aug.rl)
	{
		std::vector<unsigned char>data2(3 * x * y, 0);

		//左右反転
#pragma omp parallel for
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				int pos = (i*x + j);
				int pos2 = (i*x + x - j - 1);
				data2[pos * 3 + 0] = data[pos2 * 3 + 0];
				data2[pos * 3 + 1] = data[pos2 * 3 + 1];
				data2[pos * 3 + 2] = data[pos2 * 3 + 2];
			}
		}
		image_augmentat.push_back(data2);
	}
	if (aug.color_nize > 0.0 && aug.rnd() < aug.color_nize)
	{
		std::vector<unsigned char>data2(3 * x * y, 0);

		float c = aug.rnd();
		float rr = 1.0, gg = 1.0, bb = 1.0;
		if (c < 0.3) rr = aug.rnd();
		if (c >= 0.3 && c < 0.6) gg = aug.rnd();
		if (c >= 0.6) bb = aug.rnd();
#pragma omp parallel for
		for (int i = 0; i < x*y; i++) {
			data2[i * 3 + 0] = data[i * 3 + 0] * rr;
			data2[i * 3 + 1] = data[i * 3 + 1] * gg;
			data2[i * 3 + 2] = data[i * 3 + 2] * bb;
		}
		image_augmentat.push_back(data2);
	}

	if (aug.rnd_noize > 0.0 && aug.rnd() < aug.rnd_noize)
	{
		std::vector<unsigned char>data2(3 * x * y, 0);
#pragma omp parallel for
		for (int i = 0; i < 3 * x*y; i++) data2[i] = data[i];

		img_noize nz(15.0, aug.rnd());
		nz.noize(&data2[0], x, y);

		image_augmentat.push_back(data2);
	}
	//double g;
	if (aug.rotation > 0.0 && aug.rnd() < aug.rotation)
	{
		std::vector<unsigned char>data2(3 * x * y, 0);
#pragma omp parallel for
		for (int i = 0; i < 3 * x*y; i++) data2[i] = data[i];

		img_rotation rot;
		rot.rotation(&data2[0], x, y, (aug.rnd() < 0.5 ? 1.0 : -1.0)*(std::max(0.1,aug.rnd())*aug.rotation_max+0.5)*M_PI / 180.0);

		image_augmentat.push_back(data2);
	}

	if (aug.sift > 0.0 && aug.rnd() < aug.sift)
	{
		std::vector<unsigned char>data2(3 * x * y, 0);
#pragma omp parallel for
		for (int i = 0; i < 3 * x*y; i++) data2[i] = data[i];

		img_sift s;

		if (aug.rnd() < 0.5)
		{
			if (aug.rnd() < 0.5)
			{
				s.sift(&data2[0], x, y, 1, (int)(std::max(0.1,aug.rnd())*aug.sift_max + 0.5));
			}
			else
			{
				s.sift(&data2[0], x, y, -1, (int)(std::max(0.1, aug.rnd())*aug.sift_max + 0.5));
			}
		}
		else
		{
			if (aug.rnd() < 0.5)
			{
				s.sift(&data2[0], x, y, 2, (int)(std::max(0.1, aug.rnd())*aug.sift_max + 0.5));
			}
			else
			{
				s.sift(&data2[0], x, y, -2, (int)(std::max(0.1, aug.rnd())*aug.sift_max + 0.5));
			}
		}

		image_augmentat.push_back(data2);
	}
	return image_augmentat;
}

//#undef STB_IMAGE_IMPLEMENTATION

#endif
