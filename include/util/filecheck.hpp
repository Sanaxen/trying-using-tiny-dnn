/*
Copyright (c) 2018, Sanaxn
All rights reserved.

Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef _FILECHECK_HPP
#define _FILECHECK_HPP

#include <Windows.h>

class FileExitsCheck
{
	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind;
public:

	FileExitsCheck() { hFind = NULL; }

	inline bool isExist(std::string& filename)
	{
		hFind = FindFirstFileA(filename.c_str(), &FindFileData);
		if (hFind == INVALID_HANDLE_VALUE) 
		{
			// ë∂ç›ÇµÇ»Ç¢èÍçá
			return false;
		}
		// ë∂ç›Ç∑ÇÈèÍçá
		FindClose(hFind);
		return true;
	}

};
#endif
