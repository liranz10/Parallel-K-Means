
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

struct Point
{
	double x0;
	double y0;
	double vx;
	double vy;
	int clusterId;
};

struct Cluster
{
	int id;
	int pointsNumber;
	double xCenter;
	double yCenter;
	double sumX;
	double sumY;
	double diameter;

};


void cudaUpdatePoints(Point* updatedPoints, int pointsNumber, double t, Point* cudapointsArray);
Point* CudaPointsAllocation(Point* pointsArray, int pointsNumber);
Cluster* CudaClustersAllocation(Cluster* clustersArray, int k);
Point* cudaGroupPointsAroundClustersCenters(int numOfPoints, int k, Point* cudapointsArray, Cluster* cudaClustersArray);
void cudaFreePoints(Point* cudapointsArray);



