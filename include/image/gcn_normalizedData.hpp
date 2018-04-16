/*
Copyright (c) 2018, Sanaxn
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef _GCN_NormalizedData_HPP
#define _GCN_NormalizedData_HPP

//GCN on Normalized Data
template<class T>
inline T* gcn_normalizedData(T* data, const int dataNum, double eps = 0.0)
{
	double av = 0.0;
	const int sz = dataNum;
	T* normalized = new T[dataNum];

#pragma omp parallel for reduction(+:av)
	for (int k = 0; k < sz; k++)
	{
		av += data[k];
	}
	av /= (double)( sz);

	double sd = 0.0;
#pragma omp parallel for reduction(+:sd)
	for (int k = 0; k < sz; k++)
	{
		sd += pow(data[k] - av, 2.0);
	}
	sd = sqrt(sd / (double)(sz)) + eps;

#pragma omp parallel for
	for (int k = 0; k < sz; k++)
	{
		normalized[k] = (data[k] - av) / sd;
	}

	return normalized;
}

#endif
