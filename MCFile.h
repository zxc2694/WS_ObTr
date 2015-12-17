#ifndef MCFILE_H
#define MCFILE_H

#include <fstream>
#include <iostream>
#include <vector>

#if (WIN32)
#include <windows.h>
#endif

using namespace std;

//DWORD show_img_width = GetSystemMetrics(SM_CXSCREEN);
//DWORD show_img_height = GetSystemMetrics(SM_CYSCREEN);
const int Max_Chars_Per_Line = 1024;
const int Max_Tokens_Per_Line = 30;

bool dirExists(const std::string &dirName)
{
	DWORD attribs = GetFileAttributesA(dirName.c_str());
	if (attribs == INVALID_FILE_ATTRIBUTES){ // something wrong with the path, Use GetLastError() to find out what that failure actually is.
		return false;
	}
	else
		return (attribs & FILE_ATTRIBUTE_DIRECTORY); // this is a directory

}

bool createFolder(const char *path)
{
	if (CreateDirectoryA(path, NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		return true;
	else
		return false;
}

vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	char search_path[256];
	//wchar_t _path[256];
	sprintf_s(search_path, "%s*.*", folder.c_str());
	int dwNum = MultiByteToWideChar(CP_ACP, NULL, search_path, -1, NULL, 0);
	wchar_t *_path = new wchar_t[dwNum + 1];
	MultiByteToWideChar(CP_ACP, 0, search_path, -1, _path, dwNum);
	//
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(_path, &fd);
	//
	char filename[256];
	if (hFind != INVALID_HANDLE_VALUE){
		do{
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)){
				dwNum = WideCharToMultiByte(CP_OEMCP, NULL, fd.cFileName, -1, NULL, 0, NULL, false);
				WideCharToMultiByte(CP_OEMCP, NULL, fd.cFileName, -1, filename, dwNum, NULL, false);
				names.push_back(filename);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

int parseFile(string filename, vector<vector<float>> &roiVec)
{
	ifstream fin(filename, ios::in);
	if (!fin.is_open()){
		cout << "file is not opened." << endl;
		return -1;
	}

	int nFrames = 65536;
	int frameIdx = 0;
	bool bGetHeader = false;
	bool bGetData = false;
	char *p;
	char buf[Max_Chars_Per_Line];
	while (!fin.eof() && frameIdx < nFrames){
		fin.getline(buf, sizeof(buf));
		char *token = strtok_s(buf, " ,=", &p);
		if (strcmp(token, "!") == 0){
			bGetHeader = !bGetHeader;
			continue;
		}
		if (bGetHeader){
			if (strcmp(token, "FrameNumbers") == 0){
				nFrames = atoi(strtok_s(0, " ,=", &p));
				roiVec.resize(nFrames, vector<float>(0));
			}
		}
		if (strcmp(token,"#")==0){
			bGetData = !bGetData;
			continue;
		}
		if (bGetData){
			while (token != NULL){
				roiVec[frameIdx].push_back(atof(token));
				token = strtok_s(0, " ,", &p);
			}
			frameIdx++;
		}
			
	}
	if (frameIdx == nFrames)
		return 1;
	else
		return -2;
}


#endif