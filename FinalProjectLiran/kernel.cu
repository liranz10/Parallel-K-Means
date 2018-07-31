#include "program.h"



__global__ void UpdatePoints(Point* pointsArray, int pointsNumber, double dt)
{
	int index;
	index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < pointsNumber)
	{
		(pointsArray)[index].x0 = (pointsArray)[index].x0 + (pointsArray)[index].vx * dt;
		(pointsArray)[index].y0 = (pointsArray)[index].y0 + (pointsArray)[index].vy * dt;
	}
}


__global__  void groupPointsAroundClustersCentersKernel(Point* points, Cluster* clusters, int numOfPoints, int k)
{
	double temp = 0;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (index < numOfPoints)
	{
		double min = sqrt(pow(clusters[0].xCenter - points[index].x0, 2) + pow(clusters[0].yCenter - points[index].y0, 2));
		points[index].clusterId = 0;
		for (int j = 1; j < k; j++) {
			temp = sqrt(pow(clusters[j].xCenter - points[index].x0, 2) + pow(clusters[j].yCenter - points[index].y0, 2));
			if (temp < min)
			{
				points[index].clusterId = j;
				min = temp;
			}
		}
	}
}


void errorHanldler(cudaError_t cudaStatus, char* errorMessage, Point** pointsArrAddressOnGpu)
{
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, errorMessage, &cudaStatus, cudaGetErrorString(cudaStatus));
		cudaFree(pointsArrAddressOnGpu);

	}

}

void errorHanldler2(cudaError_t cudaStatus, char* errorMessage, Cluster** clusterArrAddressOnGpu)
{
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, errorMessage, &cudaStatus, cudaGetErrorString(cudaStatus));
		cudaFree(clusterArrAddressOnGpu);

	}

}
void cudaFreePoints(Point* cudapointsArray)
{
	cudaError_t cudaStatus;
	cudaFree(cudapointsArray);
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceReset failed!");

	}
}

void cudaUpdatePoints(Point* updatedPoints, int pointsNumber, double t, Point* cudapointsArray)
{
	cudaError_t cudaStatus;
	cudaDeviceProp deviceProperties;
	int numOfBlocks, numOfThreads;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	errorHanldler(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", &cudapointsArray);
	cudaStatus = cudaGetDeviceProperties(&deviceProperties, 0);

	numOfBlocks = (pointsNumber / deviceProperties.maxThreadsPerBlock);
	numOfThreads = deviceProperties.maxThreadsPerBlock;
	//check if num of blocks is not enough!
	if ((pointsNumber % deviceProperties.maxThreadsPerBlock) != 0)
		numOfBlocks++;
	// Launch a kernel on the GPU 
	UpdatePoints << <numOfBlocks, numOfThreads >> > (cudapointsArray, pointsNumber, t);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	errorHanldler(cudaStatus, "UpdatePoints Kernel launch failed : %s\n", &cudapointsArray);
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	errorHanldler(cudaStatus, "cudaDeviceSynchronize returned error code %d after launching UpdatePoints Kernel!\n", &cudapointsArray);



	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(updatedPoints, cudapointsArray, pointsNumber * sizeof(Point), cudaMemcpyDeviceToHost);
	errorHanldler(cudaStatus, "cudaMemcpy failed!\n", &cudapointsArray);

}
Point* cudaGroupPointsAroundClustersCenters(int numOfPoints, int k, Point* cudapointsArray,Cluster* cudaClustersArray)
{
	cudaError_t cudaStatus;
	cudaDeviceProp deviceProperties;
	int numOfBlocks, numOfThreads;
	Point* updatedPoints = (Point*)malloc(numOfPoints*sizeof(Point));

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	errorHanldler(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", &cudapointsArray);
	cudaStatus = cudaGetDeviceProperties(&deviceProperties, 0);

	numOfBlocks = (numOfPoints / deviceProperties.maxThreadsPerBlock);
	numOfThreads = deviceProperties.maxThreadsPerBlock;
	//check if num of blocks is not enough!
	if ((numOfPoints % deviceProperties.maxThreadsPerBlock) != 0)
		numOfBlocks++;
	// Launch a kernel on the GPU 
	//printPointsKernel << <numOfBlocks, numOfThreads >> > (cudapointsArray, numOfPoints);

	groupPointsAroundClustersCentersKernel << <numOfBlocks, numOfThreads >> > (cudapointsArray, cudaClustersArray, numOfPoints,k);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	errorHanldler(cudaStatus, "GroupPointsAroundClustersCenters Kernel launch failed : %s\n", &cudapointsArray);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	errorHanldler(cudaStatus, "cudaDeviceSynchronize returned error code %d after launching groupPointsAroundClustersCenters Kernel!\n", &cudapointsArray);


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(updatedPoints, cudapointsArray, numOfPoints * sizeof(Point), cudaMemcpyDeviceToHost);
	errorHanldler(cudaStatus, "cudaMemcpy failed!\n", &cudapointsArray);
	cudaFree(cudaClustersArray);



	return updatedPoints;
}

Point* CudaPointsAllocation(Point* pointsArray, int pointsNumber)
{
	Point *pointsArrAddressOnGpu = 0;


	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	errorHanldler(cudaStatus, "cudaSetDevice PointsAllocation failed!   Do you have a CUDA-capable GPU installed?\n", &pointsArrAddressOnGpu);


	// Allocate GPU buffers for two arrays (one input, one output)    .
	cudaStatus = cudaMalloc((void**)&pointsArrAddressOnGpu, pointsNumber * sizeof(Point));
	errorHanldler(cudaStatus, "cudaMalloc PointsAllocation failed!\n", &pointsArrAddressOnGpu);


	// Copy input array from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(pointsArrAddressOnGpu, pointsArray, pointsNumber * sizeof(Point), cudaMemcpyHostToDevice);
	errorHanldler(cudaStatus, "cudaMemcpy PointsAllocation failed!\n", &pointsArrAddressOnGpu);

	return pointsArrAddressOnGpu;

}


Cluster* CudaClustersAllocation(Cluster* clustersArray, int k)
{
	Cluster *clustersArrayAddressOnGpu = 0;


	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	errorHanldler2(cudaStatus, "cudaSetDevice ClustersAllocation failed!   Do you have a CUDA-capable GPU installed?\n", &clustersArrayAddressOnGpu);


	// Allocate GPU buffers for two arrays (one input, one output)    .
	cudaStatus = cudaMalloc((void**)&clustersArrayAddressOnGpu, k * sizeof(Cluster));
	errorHanldler2(cudaStatus, "cudaMalloc ClustersAllocation failed!\n", &clustersArrayAddressOnGpu);


	// Copy input array from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(clustersArrayAddressOnGpu, clustersArray, k * sizeof(Cluster), cudaMemcpyHostToDevice);
	errorHanldler2(cudaStatus, "cudaMemcpy ClustersAllocation failed!\n", &clustersArrayAddressOnGpu);

	return clustersArrayAddressOnGpu;

}

