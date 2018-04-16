/*
Copyright (c) 2018, Sanaxn
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef _DirectryTool_h
#define _DirectryTool_h

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#include "windows.h"

#include <imagehlp.h>					//MakeSureDirectoryPathExists
#pragma comment(lib, "imagehlp.lib")

#include <shlwapi.h>					//PathIsDirectoryA
#pragma comment(lib, "shlwapi.lib")


class DirectryTool
{
public:
	bool ExistDir(char* dirname)
	{
		return (bool)PathIsDirectoryA(dirname);

	}

	bool MakeDir(char* dirname)
	{
		if ( ExistDir(dirname) )
		{
			return true;
		}
		return (bool)MakeSureDirectoryPathExists(dirname);
	}
};

#endif