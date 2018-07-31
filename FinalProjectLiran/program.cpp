#include <time.h>
#include <stdlib.h>
#include "program.h"




void updatePoints(Point* pointsArray, int pointsNumber, double t)
{
	#pragma omp parallel for
	for (int i = 0; i < pointsNumber; i++)
	{
		(pointsArray)[i].x0 = (pointsArray)[i].x0 + (pointsArray)[i].vx * t;
		(pointsArray)[i].y0 = (pointsArray)[i].y0 + (pointsArray)[i].vy * t;
	}

}

void printPoints(Point* pointsArray, int pointsNumber)
{
	for (int i = 0; i < pointsNumber; i++)
	{
		printf("\n x: %f  y: %f  cid: %d", pointsArray[i].x0, pointsArray[i].y0, pointsArray[i].clusterId);
	}
	fflush(stdout);
}

void printClusters(Cluster* clusters, int k)
{
	for (int i = 0; i < k; i++)
	{
		printf("\n id: %d xCenter: %f  yCenter: %f points number: %d diameter: %f", clusters[i].id, clusters[i].xCenter, clusters[i].yCenter, clusters[i].pointsNumber, clusters[i].diameter);
	}
	fflush(stdout);
}

void ReadPointsFromFile(Point** pointsArray, char* filename, int* pointsNumber, int* numberOfClustersTofind, double* time, double* deltaT, double* limit, double* qualityMeasure)
{
	int i;
	FILE *f = fopen(filename, "r");
	if (f == NULL)
	{
		printf("Error Reading File!");
		return;
	}
	fscanf(f, "%d %d %lf %lf %lf %lf", pointsNumber, numberOfClustersTofind, time, deltaT, limit, qualityMeasure);
	*pointsArray = (Point*)malloc(*pointsNumber * sizeof(Point));
	for (i = 0; i < *pointsNumber; i++)
	{
		float x, y, vx, vy;
		fscanf(f, "%f %f %f %f", &x, &y, &vx, &vy);
		(*pointsArray)[i].x0 = x;
		(*pointsArray)[i].y0 = y;
		(*pointsArray)[i].vx = vx;
		(*pointsArray)[i].vy = vy;
		(*pointsArray)[i].clusterId = -1;
		fflush(stdout);
	}

}

void ZeroingAllSums(Cluster* clusters, int k)
{
	#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		(clusters)[i].sumX = 0;
		(clusters)[i].sumY = 0;
	}
}

void chooseFirstKClustersCenter(Point* points, int numOfPoints, int k, Cluster* clusters)
{

	#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		clusters[i].xCenter = (points)[i].x0;
		clusters[i].yCenter = (points)[i].y0;
		clusters[i].pointsNumber = 0;
		clusters[i].id = i;
		clusters[i].diameter = 0;
	}
	
}
double CalculateDistanceFromCenter(Point p, double centerX, double centerY)
{
	return sqrt(pow(centerX - p.x0, 2) + pow(centerY - p.y0, 2));
}
double CalculateDistance(Point p1, Point p2)
{
	return sqrt(pow(p1.x0 - p2.x0, 2) + pow(p1.y0 - p2.y0, 2));
}

void addPointToCluster(Cluster *cluster, Point* point, Cluster *clusters, int* PointsMovedToOtherClusters)
{

	if ((*point).clusterId != (*cluster).id || (*point).clusterId == -1)
	{
		*PointsMovedToOtherClusters = 1;
		(*point).clusterId = (*cluster).id;
	}
}

void groupPointsAroundClustersCenters(Point* points, Cluster* clusters, int numOfPoints, int k, int* PointsMovedToOtherClusters)
{
	#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		(clusters)[i].pointsNumber = 0;
	}
	ZeroingAllSums(clusters, k);


	double temp = 0;
	for (int i = 0; i < numOfPoints; i++)
	{
		double min = CalculateDistanceFromCenter((points)[i], (clusters)[0].xCenter, (clusters)[0].yCenter);
		int id = 0;
		for (int j = 1; j < k; j++) {
			temp = CalculateDistanceFromCenter((points)[i], (clusters)[j].xCenter, (clusters)[j].yCenter);
			if (temp < min)
			{
				id = j;
				min = temp;
			}
		}
		addPointToCluster(&(clusters)[id], &(points)[i], clusters, PointsMovedToOtherClusters);
	}

	for (int i = 0; i < numOfPoints; i++)
	{
		clusters[points[i].clusterId].pointsNumber++;
		clusters[(points)[i].clusterId].sumX += (points)[i].x0;
		clusters[(points)[i].clusterId].sumY += (points)[i].y0;
	}


}

void GroupPointsAroundClustersCentersHandler(Point* points, Cluster* clusters, int numOfPoints, int k, Point* cudaPointsArr, Cluster* cudaClusters, int* PointsMovedToOtherClusters)
{
	Point* cudaUpdatedPoints =cudaGroupPointsAroundClustersCenters(numOfPoints, k, cudaPointsArr, cudaClusters);
	//reset clusters
	#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		(clusters)[i].pointsNumber = 0;
	}
	ZeroingAllSums(clusters, k);
	
	//update cpu points
	for (int i = 0; i < numOfPoints; i++)
	{
		if (cudaUpdatedPoints[i].clusterId != points[i].clusterId || points[i].clusterId == -1)
		{
			points[i].clusterId = cudaUpdatedPoints[i].clusterId;
			clusters[cudaUpdatedPoints[i].clusterId].pointsNumber++;
			clusters[cudaUpdatedPoints[i].clusterId].sumX+= cudaUpdatedPoints[i].x0;
			clusters[cudaUpdatedPoints[i].clusterId].sumY += cudaUpdatedPoints[i].y0;
			*PointsMovedToOtherClusters = 1;
		}
		else
		{
			clusters[cudaUpdatedPoints[i].clusterId].pointsNumber++;
			clusters[cudaUpdatedPoints[i].clusterId].sumX += cudaUpdatedPoints[i].x0;
			clusters[cudaUpdatedPoints[i].clusterId].sumY += cudaUpdatedPoints[i].y0;
		}
	}
	free(cudaUpdatedPoints);

}

void recalculateClustersCenters(Cluster* clusters, int k)
{
	for (int i = 0; i < k; i++)
	{
		(clusters)[i].xCenter =	(clusters[i].sumX) / (clusters[i].pointsNumber);
		(clusters)[i].yCenter = (clusters[i].sumY) / (clusters[i].pointsNumber);
	}
}


double CalculateDistanceBetweenClusters(Cluster c1, Cluster c2)
{
	return sqrt(pow(c1.xCenter - c2.xCenter, 2) + pow(c1.yCenter - c2.yCenter, 2));
}

void CalculateClustersDiameters(Cluster* clusters, Point* points, int numOfPoints, int k)
{
	double* threadsClustersArray = (double*)calloc(k*omp_get_max_threads(), sizeof(double));
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int i = 0; i < numOfPoints; i++)
	{
		int tid = omp_get_thread_num();
		for (int j = i + 1; j < numOfPoints; j++)
		{
			if (points[i].clusterId == points[j].clusterId)
			{
				double temp = CalculateDistance(points[i], points[j]);
				if (threadsClustersArray[tid * k + points[i].clusterId] < temp)
					threadsClustersArray[tid * k + points[i].clusterId] = temp;
			}
		}
	}

	//merge : find max in threadsClustersArray
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < omp_get_max_threads(); j++)
		{
			if (clusters[i].diameter < threadsClustersArray[i + j*k])
				clusters[i].diameter = threadsClustersArray[i + j*k];
		}
	}
}


void EvaluateQualityOfClusters(Cluster* clusters, int k, Point* points, int numOfPoints, double* qm)
{
	CalculateClustersDiameters(clusters, points, numOfPoints, k);
	double sum = 0;
	*qm = 0;
	//check if threads num is less then k
	int resultArrSize = k < omp_get_max_threads() ? k : omp_get_max_threads();
	double* resultArr = (double*)malloc(resultArrSize * sizeof(double));
	for (int i = 0; i < resultArrSize; i++)
	{
		resultArr[i] = 0;
	}
#pragma omp parallel for
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < k; j++)
		{
			if (j != i)
			{
				resultArr[omp_get_thread_num()] += clusters[i].diameter / CalculateDistanceBetweenClusters(clusters[i], clusters[j]);
			}
		}
	}
	//merging
	for (int i = 0; i < resultArrSize; i++)
	{
		sum += resultArr[i];
	}

	*qm = sum / (k*(k - 1));
}



void definePointType(MPI_Datatype *PointType)
{
	Point point;
	MPI_Datatype type[5] = { MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_INT };
	int blocklen[5] = { 1,1,1,1,1 };
	MPI_Aint disp[5];
	disp[0] = (char *)&point.x0 - (char *)&point;
	disp[1] = (char *)&point.y0 - (char *)&point;
	disp[2] = (char *)&point.vx - (char *)&point;
	disp[3] = (char *)&point.vy - (char *)&point;
	disp[4] = (char *)&point.clusterId - (char *)&point;
	MPI_Type_create_struct(5, blocklen, disp, type, PointType);
	MPI_Type_commit(PointType);
}

void defineClusterType(MPI_Datatype *ClusterType)
{
	Cluster cluster;
	MPI_Datatype type[7] = { MPI_INT,MPI_INT,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE };
	int blocklen[7] = { 1,1,1,1,1,1,1 };
	MPI_Aint disp[7];
	disp[0] = (char *)&cluster.id - (char *)&cluster;
	disp[1] = (char *)&cluster.pointsNumber - (char *)&cluster;
	disp[2] = (char *)&cluster.xCenter - (char *)&cluster;
	disp[3] = (char *)&cluster.yCenter - (char *)&cluster;
	disp[4] = (char *)&cluster.sumX - (char *)&cluster;
	disp[5] = (char *)&cluster.sumY - (char *)&cluster;
	disp[6] = (char *)&cluster.diameter - (char *)&cluster;

	MPI_Type_create_struct(7, blocklen, disp, type, ClusterType);
	MPI_Type_commit(ClusterType);
}



void broadcastKmeanParmeters(int k, double limit, double qm, int numOfPoints,double time, double dt,int pointsPerProcess)
{
	MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&limit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&qm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&numOfPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&pointsPerProcess, 1, MPI_INT, 0, MPI_COMM_WORLD);


}


void tasksDistribution(Point* points, MPI_Datatype PointType, int numOfProcesses, int pointsPerProcess, Point* mypoints)
{
	MPI_Scatter(points, pointsPerProcess, PointType, mypoints, pointsPerProcess, PointType, 0, MPI_COMM_WORLD);
}


void sendClustersToMaster(Cluster *clusters, MPI_Datatype ClusterType, int k)
{
	MPI_Send(clusters, k, ClusterType, 0, 0, MPI_COMM_WORLD);
}

void getClustersResultsFromSlaves(Cluster *clusters, MPI_Datatype ClusterType, int numOfProcesses, int k, MPI_Status *status)
{

	Cluster* tempClusters;
	tempClusters = (Cluster*)calloc(k, sizeof(Cluster));
	for (int i = 1; i < numOfProcesses; i++)
	{
		MPI_Recv(tempClusters, k, ClusterType, i, 0, MPI_COMM_WORLD, status);
		for (int j = 0; j < k; j++)
		{
			clusters[j].pointsNumber += tempClusters[j].pointsNumber;
			clusters[j].sumX += tempClusters[j].sumX;
			clusters[j].sumY += tempClusters[j].sumY;
		}
	}
	free(tempClusters);
}

int checkAllFlags(int numOfProcesses,int* flags)
{

	int flag = 0;

	for (int i = 0; i < numOfProcesses; i++)
	{
		if (flags[i] == 1)
			flag = 1;
	}
	return flag;

}
void writeToFile(Cluster* clusters, int k, double q, double t) {
	FILE* f = fopen("D:\\Liran\\output.txt", "w");
	fprintf(f, "First occurrence at t = %f with q = %f\n", t, q);
	fprintf(f, "Centers of the clusters:\n");
	for (int i = 0; i < k; i++)
	{
		fprintf(f, "%f    %f\n", clusters[i].xCenter, clusters[i].yCenter);
	}
}


int main(int argc, char *argv[])
{
	int myrank, numOfProcesses;
	MPI_Datatype PointType;
	MPI_Datatype ClusterType;
	Point* points = NULL;
	Point* mypoints = NULL;
	Cluster* clusters = NULL;
	int k, numOfPoints;
	double dt, qM, q = 0, t = 0, start,finish,limit,time;
	char* filename = "D:\\Liran\\input.txt";
	

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	definePointType(&PointType);
	defineClusterType(&ClusterType);
	int PointsMovedToOtherClusters = 0;
	int pointsPerProcess;

	Point* cudaPointsArr = NULL;
	Cluster* cudaClusters = NULL;
	int* flagsArr = (int*)calloc(numOfProcesses, sizeof(int));
	start = omp_get_wtime();
	
	//master read points and info from file ans sends it to slaves
	if (myrank == 0)
	{
		ReadPointsFromFile(&points, filename, &numOfPoints, &k, &time, &dt, &limit, &qM);
		clusters = (Cluster*)malloc(k * sizeof(Cluster));
		pointsPerProcess = (numOfPoints / numOfProcesses);
		broadcastKmeanParmeters(k, limit, qM, numOfPoints,time,dt, pointsPerProcess);
		mypoints = (Point*)calloc(pointsPerProcess, sizeof(Point));
		//divide points between processes
		tasksDistribution(points, PointType, numOfProcesses, pointsPerProcess, mypoints);
		chooseFirstKClustersCenter(points, numOfPoints, k, clusters);
		//allocate points at gpu
		cudaPointsArr = CudaPointsAllocation(mypoints, pointsPerProcess);
	}
	else
	{	
		MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&limit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&qM, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&numOfPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&pointsPerProcess, 1, MPI_INT, 0, MPI_COMM_WORLD);
		clusters = (Cluster*)malloc(k * sizeof(Cluster));
		mypoints = (Point*)calloc(pointsPerProcess, sizeof(Point));
		tasksDistribution(points, PointType, numOfProcesses, pointsPerProcess, mypoints);
		//allocate points at gpu
		cudaPointsArr = CudaPointsAllocation(mypoints, pointsPerProcess);
	}
	do
	{
		//k means algorithem
		do
		{
			//share clusters
			MPI_Bcast(clusters, k, ClusterType, 0, MPI_COMM_WORLD);
			
			PointsMovedToOtherClusters = 0;
			//gpu clusters allocate
			cudaClusters = CudaClustersAllocation(clusters, k);
			if (myrank == 0)
			{
				GroupPointsAroundClustersCentersHandler(mypoints, clusters, pointsPerProcess, k, cudaPointsArr, cudaClusters, &PointsMovedToOtherClusters);
				//get flags from slaves
				MPI_Gather(&PointsMovedToOtherClusters, 1, MPI_INT, flagsArr, numOfProcesses, MPI_INT, 0, MPI_COMM_WORLD);
				//master checks all slaves flags
				PointsMovedToOtherClusters = checkAllFlags(numOfProcesses, flagsArr);
				//send global flag to slaves
				MPI_Bcast(&PointsMovedToOtherClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);

				getClustersResultsFromSlaves(clusters, ClusterType, numOfProcesses, k, &status);
				recalculateClustersCenters(clusters, k);
			}
			else
			{
				GroupPointsAroundClustersCentersHandler(mypoints, clusters, pointsPerProcess, k, cudaPointsArr, cudaClusters, &PointsMovedToOtherClusters);
				//send flag to master
				MPI_Gather(&PointsMovedToOtherClusters, 1, MPI_INT, flagsArr, numOfProcesses, MPI_INT, 0, MPI_COMM_WORLD);
				//get global flag from master
				MPI_Bcast(&PointsMovedToOtherClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
				sendClustersToMaster(clusters, ClusterType, k);
				
			}
			limit--;
		} while (limit > 0 && PointsMovedToOtherClusters);
		
		//gather all points
		MPI_Gather(mypoints, pointsPerProcess, PointType, points,pointsPerProcess,  PointType, 0, MPI_COMM_WORLD);

		//master evaluate quality
		if (myrank == 0)
		{
			EvaluateQualityOfClusters(clusters, k, points, numOfPoints, &q);
			//send q to slaves
			for (int i = 1; i < numOfProcesses; i++)
			{
				MPI_Send(&q, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
			if (q < qM)
				break;
		}
		else
		{	
			//recieve q from master
			MPI_Recv(&q, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
			if (q < qM)
				break;
		}
		
		//update points
		if (myrank == 0)
		{
			cudaUpdatePoints(mypoints, pointsPerProcess, t, cudaPointsArr);
		}
		else
			cudaUpdatePoints(mypoints, pointsPerProcess, t, cudaPointsArr);


		t += dt;
		
	} while (t < time && q > qM);


	if (myrank == 0)
	{
		finish = omp_get_wtime();
		writeToFile(clusters, k, q, t);
		printClusters(clusters, k);
		printf("\n%f", q);
		fflush(stdout);
		printf("\n%f", (finish - start));
		fflush(stdout);
		free(points);
	}
	cudaFreePoints(cudaPointsArr);
	free(clusters);
	free(mypoints);
	MPI_Finalize();

}


