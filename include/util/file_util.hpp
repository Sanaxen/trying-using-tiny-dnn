/*
Copyright (c) 2018, Sanaxn
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef _FILE_UTIL_HPP
#define _FILE_UTIL_HPP

#ifndef _Windows
#include <Windows.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

inline FILE* fopen_util(const char* filename, const char* mode)
{
	FILE* fp = NULL;
	
	for ( int k = 0; k < 1000; k++ )
	{
		fp = fopen(filename, mode);
		if ( fp ) break;
		printf("open [%s] waiting!! %d/%d\n", filename, k+1, 1000);
#ifndef _Windows
		Sleep(100);
#else
		sleep(100);
#endif
	}

	return fp;
}

#endif
