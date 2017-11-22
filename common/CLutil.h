#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define     ERROR_LEN                (1024)
//#define 

char BufferError[ERROR_LEN];
cl_int status;
cl_context g_context;
cl_command_queue g_queue;
cl_program program;


void checkErr(cl_int err,const char *name)
{
	if(err != CL_SUCCESS)
	{
		printf("Error: %s %d",name,err);
		switch(err)
		{
			case CL_DEVICE_NOT_FOUND :printf("(CL_DEVICE_NOT_FOUND)");break;
			case CL_DEVICE_NOT_AVAILABLE :printf("(CL_DEVICE_NOT_AVAILABLE)");break;
			case CL_COMPILER_NOT_AVAILABLE :printf("(CL_COMPILER_NOT_AVAILABLE)");break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE :printf("(CL_MEM_OBJECT_ALIOCATION_FAILURE)");break;
			case CL_OUT_OF_RESOURCES :printf("(CL_OUT_OF_RESOURCES)");break;
			case CL_OUT_OF_HOST_MEMORY :printf("(CL_OUT_OF_HOST_MEMORY)");break;
			case CL_MEM_COPY_OVERLAP :printf("(CL_MEM_COPY_OVERLAP)");break;
			case CL_BUILD_PROGRAM_FAILURE:printf("(CL_BUILD_PROGRAM_FAILURE)");break;
			case CL_INVALID_VALUE:printf("(CL_INVALID_VALUE)");break;
			case CL_INVALID_DEVICE_TYPE:printf("(CL_INVALID_DEVICE_TYPE)");break;
			case CL_INVALID_DEVICE:printf("(CL_INVALID_DEVICE)");break;
			case CL_INVALID_CONTEXT:printf("(CL_INVALID_CONTEXT)");break;
			case CL_INVALID_BINARY:printf("(CL_INVALID_BINARY)");break;
			case CL_INVALID_BUILD_OPTIONS:printf("(CL_INVALID_BUILD_OPTIONS)");break;
			case CL_INVALID_PROGRAM:printf("(CL_INVALID_PROGRAM)");break;
			case CL_INVALID_PROGRAM_EXECUTABLE:printf("(CL_INVALID_PROGRAM_EXECUTABLE)");break;
			case CL_INVALID_KERNEL_DEFINITION:printf("(CL_INVALID_KERNEL_DEFINITION)");break;
			case CL_INVALID_KERNEL:printf("(CL_INVALID_KERNEL)");break;
			case CL_INVALID_KERNEL_ARGS:printf("(CL_INVALID_KERNEL_ARGS)");break;
			case CL_INVALID_OPERATION:printf("(CL_INVALID_OPERATION)");break;
			case CL_INVALID_COMMAND_QUEUE:printf("(CL_INVALID_COMMAND_QUEUE)");break;
			case CL_INVALID_WORK_DIMENSION:printf("(CL_INVALID_WORK_DIMENSION)");break;
			case CL_INVALID_WORK_GROUP_SIZE:printf("(CL_INVALID_WORK_GROUP_SIZE)");break;
			case CL_INVALID_WORK_ITEM_SIZE:printf("(CL_INVALID_WORK_ITEM_SIZE)");break;
			case CL_INVALID_GLOBAL_WORK_SIZE:printf("(CL_INVALID_GLOBAL_WORK_SIZE)");break;
			case CL_INVALID_GLOBAL_OFFSET:printf("(CL_INVALID_GLOBAL_OFFSET)");break;
			case CL_INVALID_IMAGE_SIZE:printf("(CL_INVALID_IMAGE_SIZE)");break;
			case CL_INVALID_EVENT_WAIT_LIST:printf("(CL_INVALID_EVENT_WAIT_LIST)");break;
			case CL_MISALIGNED_SUB_BUFFER_OFFSET:printf("(CL_MISALIGNED_SUB_BUFFER_OFFSET)");break;

			default:
												 break;
		}
		printf("\n");
		exit(1);
	}
}

char* ReadSources(const char* fileName)
{
	FILE *file=fopen(fileName,"rb");
	if(!file)
	{
		printf("ERROR:Failed to open file '%s'\n",fileName);
		return NULL;
	}

	if(fseek(file,0,SEEK_END))
	{
		printf("ERROR: Failed to open file '%s'\n",fileName);
		fclose(file);
		return NULL;
	}

	long size=ftell(file);
	if(size==0)
	{
		printf("ERROR: Failed to check position on file '%s'\n",fileName);
		fclose(file);
		return NULL;
	}

	rewind(file);

	char *src=(char *)malloc(sizeof(char)*size+1);
	if(!src)
	{
		printf("ERROR: Failed to allocate memory for file '%s'\n",fileName);
		fclose(file);
		return NULL;
	}

	size_t res=fread(src,1,sizeof(char) *size,file);
	if(res !=sizeof (char) * size)
	{
		printf("ERROR: Failed to read file '%s'\n",fileName);
		fclose(file);
		free(src);
		return NULL;
	}

	src[size]='\0';/*NULL terminated */
	fclose(file);

	return src;
}

int openCLCreate(cl_context* context,cl_command_queue* queue,const char* inputfile)
{
	cl_int status;
	cl_platform_id platform=NULL;
	cl_device_id *device;
	cl_uint numPlatforms,numdevices;

	status=clGetPlatformIDs(0,NULL,&numPlatforms);
	checkErr(status,"clGetPlatformIDs()");
	if(numPlatforms>0)
	{
		cl_platform_id*  platforms=(cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
		status =clGetPlatformIDs(numPlatforms,platforms,NULL);
		if(status !=CL_SUCCESS)
		{
			printf("Error:Getting platform Ids.(clGetPlatformsIDs)\n");
			return -1;
		}
		for(unsigned int i=0; i<numPlatforms;++i)
		{
			char pbuff[100];
			status = clGetPlatformInfo(
					platforms[i],
					CL_PLATFORM_VENDOR,
					sizeof(pbuff),
					pbuff,
					NULL
					);
			platform=platforms[i];
			if(!strcmp(pbuff,"Advanced Micro Devices,Inc."))
			{
				break;
			} 
		}
		delete platforms;
	}

	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,NULL,&numdevices);
	checkErr(status,"clGetDeviceIDs()");

	device=(cl_device_id *)malloc(numdevices*sizeof(cl_device_id));
	status=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,numdevices,device,NULL);

	*context=clCreateContext(NULL,numdevices,device,NULL,NULL,&status);
	checkErr(status,"clCreateContext()");

	*queue=clCreateCommandQueue(*context,device[1],CL_QUEUE_PROFILING_ENABLE,&status);
	checkErr(status,"clCreateCommandQueue()");

	char *program_source=ReadSources((const char*)inputfile);

	program=clCreateProgramWithSource(g_context,1,(const char**)&program_source,NULL,&status);
	checkErr(status,"clCreateProgramWithSource");

	status=clBuildProgram(program,1,&device[1],NULL,NULL,NULL);

	if(status !=CL_SUCCESS)
	{
		printf("Error: build kernel fails\n");
		size_t len;
		clGetProgramBuildInfo(program,*device,CL_PROGRAM_BUILD_LOG,sizeof(BufferError),BufferError,&len);
		BufferError[len]='\0';
		printf("%s",BufferError);
		checkErr(status,"clBuildProgram");
		getchar();
		return -1;
	}

	return 0;
}
