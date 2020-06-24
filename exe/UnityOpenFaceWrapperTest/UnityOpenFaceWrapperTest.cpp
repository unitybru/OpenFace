// UnityOpenFaceWrapperTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <windows.h>
#include <iostream>

// TODO: put this in the same file as the declarations in the DLL to avoid mismatches
extern "C" __declspec(dllimport) bool OpenFaceSetup(const char* executablePath);
extern "C" __declspec(dllimport) std::string OpenFaceGetFeatures(const char* pixels, int width, int height);
extern "C" __declspec(dllimport) bool OpenFaceClose();

int main()
{
    std::cout << "Hello World!\n";

    std::string sPath = "C:\\DEV\\HACKWEEK\\OpenFaceUnity\\x64\\Debug\\FaceLandmarkImg.exe";

    // Initialize OpenFace wrapper
    OpenFaceSetup(sPath.c_str());

    char test[640*480*3]; // 3 bytes per pixel
    for (int i = 0; i < 640*480*3; i++)
        test[i] = 0xff;
    std::string str = OpenFaceGetFeatures(test, 640, 480);

    // Close OpenFace wrapper
    OpenFaceClose();

    // Test
    OpenFaceSetup(sPath.c_str());
    OpenFaceClose();
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
