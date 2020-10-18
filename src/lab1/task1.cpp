#include <cstdio>
#include <CL/cl.h>

int main(int argc, char *argv[]) {
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    cl_platform_id *platform = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platform, nullptr);

    for (cl_uint i = 0; i < platformCount; ++i) {
        char platformName[128];
        clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
        printf("Platform: %s\n", platformName);

        cl_uint deviceCount;
        clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        cl_device_id *device = (cl_device_id *) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, deviceCount, device, nullptr);

        for (cl_uint i = 0; i < deviceCount; ++i) {
            char deviceName[128];
            clGetDeviceInfo(device[i], CL_DEVICE_NAME, 128, deviceName, nullptr);
            printf("Device: %s\n", deviceName);
        }
    }
    return 0;
}
