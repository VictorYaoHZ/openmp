#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<sys/time.h>
#include <iostream>
#include <omp.h>
#include<limits.h>
using namespace std;
#define N 2048//点的个数
#define K 8//聚类的个数
#define E 256 //元素的个数
#define NUM_THREAD 8
#define MAX_ITER 20
typedef struct
{
	float elements[E];
}Point;
Point tep[K];
Point mean[K];  ///  保存每个簇的中心点
int count1[K];
int center[N];  ///  判断每个点属于哪个簇 center[k]=p，即第k个point位于第p聚类

pthread_barrier_t	barrier1;
//Point point[N];
float minn;
float d=0.0;
float cnt = 0.0;//quanjubianliang!
float global_sum=0;
int nowindex=0;
Point point[N] = {
	{1,1,1,1},
	{1,2,1,1},
	{2,1,2,1},
	{2,2,1,1},
	{50,49,49,50},
	{50,51,51,50},
	{51,49,51,49},
	{100,99,98,101},
	{100,101,100,100},
	{101,99,98,100},
	{102,99,101,100},
	{98,101,101,98}
};


void printPointInfo(int index)
{
	cout << "点 :(";
	cout << point[index].elements[0];
	for (int i = 1; i < E; i++)
	{
		cout << "," << point[index].elements[i];
	}
	cout << ") 在聚类" << center[index] + 1 << "中" << endl;
}
void printCenterInfo(int index)
{
	cout << "聚类" << index + 1 << "的新中心点是:(";
	cout << mean[index].elements[0];
	for (int i = 1; i < E; i++)
	{
		cout << "," << mean[index].elements[i];
	}
	cout << ")" << endl;
}
float getDistance(Point point1, Point point2)//计算欧氏距离
{
	float d = 0.0;
	for (int i = 0; i < E; i++)
	{
		d += (point1.elements[i] - point2.elements[i]) * (point1.elements[i] - point2.elements[i]);
	}
	d = sqrt(d);
	return d;
}
void cluster()
{

	float min;
	for (int i = 0; i < N; ++i)
	{
		min = (float)INT_MAX;
		for (int j = 0; j < K; ++j)
		{
			//distance[i][j] = getDistance(point[i], mean[j]);

			float d = 0.0;
			for (int k = 0; k < E; k++)
			{
				d += (point[i].elements[k] - mean[j].elements[k]) * (point[i].elements[k] - mean[j].elements[k]);
			}
			//d = sqrt(d);
			//***********************************************
			// 这里发现不需要用一个distance数组来进行记录，最后再顺序比较数组内数据
			// 只需要用一个临时变量来代替，顺序比较更新欧氏距离最小的聚类中心点即可
			//***********************************************
			if (d < min)
			{
				min = d;
				center[i] = j;
			}
			//distance[i][j] = d;
			/// printf("%f\n", distance[i][j]);  /// 可以用来测试对于每个点与3个中心点之间的距离
		}

		//printPointInfo(i);
	}
	//printf("-----------------------------\n");
}
float getE()
{
	int i, j, k;
	float cnt = 0.0, sum = 0.0;
	for (i = 0; i < K; ++i)
	{
		for (j = 0; j < N; ++j)
		{

			if (i == center[j])
			{
				//cnt = (point[j].x - mean[i].x) * (point[j].x - mean[i].x) + (point[j].y - mean[i].y) * (point[j].y - mean[i].y);
				for (k = 0; k < E; k++)
				{
					cnt += (point[j].elements[k] - mean[i].elements[k]) * (point[j].elements[k] - mean[i].elements[k]);
				}
				sum += cnt;
				cnt=0;
			}
		}
	}
	return sum;
}
//--------------------------
void kmeans()
{
    //================================================
    //重置count1与tep
	for (int i = 0; i < K; i++) count1[i] = 0;
    for (int m = 0; m < K; ++m)
    {
        for (int n = 0; n < E; ++n)
        {
            tep[m].elements[n] = 0;
        }
    }
    //================================================
    //根据上一轮聚类的结果，重新计算聚类中心，原getMean函数
    //累加
    for (int j = 0; j < N; ++j)
    {
        count1[center[j]]++;
        for (int m = 0; m < E; m++)
        {
            tep[center[j]].elements[m] += point[j].elements[m] * 100000;
        }
    }
    //求平均
    for (int i = 0; i < K; i++)
    {
        for (int m = 0; m < E; ++m)
        {
            mean[i].elements[m] = tep[i].elements[m] / (count1[i] * 100000);
        }
    }
    //==========================================
    //根据新的聚类中心，重新进行聚类，原cluster函数
	for (int i = 0; i < N; ++i)
	{
		minn = 99999999;
		for (int j = 0; j < K; ++j)
		{
			d = 0.0;
			for (int k = 0; k < E; k++)
			{
				d += (point[i].elements[k] - mean[j].elements[k]) * (point[i].elements[k] - mean[j].elements[k]);
			}
			if (d < minn)
			{
				minn = d;
				center[i] = j;
			}
		}
		//printPointInfo(i);
	}

}
//========================================================================================================

//dynamic?
//直接将dynamic替换static不行
//由并行区域放到pragma omp for不行
//重新单独在要进行dynamic schedule的循环区域建立并行区域，声明好private变量可以
void kmeans_openMP()
{
    #pragma omp parallel num_threads(NUM_THREAD),shared(tep,count1,center,mean,point)
    for(int iter_num=1;iter_num<MAX_ITER;iter_num++)
    {
         //================================================
        //重置count1与tep
		#pragma omp master
		{
        for (int i = 0; i < K; i++) count1[i] = 0;}
	#pragma omp for
        for (int m = 0; m < K; ++m)
        {
	//#pragma omp simd
            for (int n = 0; n < E; ++n)
            {
                tep[m].elements[n] = 0;
            }
        }
	
	#pragma omp barrier
        //================================================
        //根据上一轮聚类的结果，重新计算聚类中心，原getMean函数
        //累加
        //#pragma omp for schedule(static, N/NUM_THREAD)
	//#pragma omp for private(nowindex),schedule(dynamic,1)
	 #pragma omp single 
        for (int j = 0; j < N; ++j)
        {
	    nowindex = center[j];
            count1[nowindex]++;
	//	#pragma omp simd
            for (int m = 0; m < E; m++)
            {	
		
                tep[nowindex].elements[m] += point[j].elements[m]*100000;
		//cout<<j<<"  "<<m<<" "<<nowindex<<endl;
            }
        }
	
        //求平均
	#pragma omp barrier
        #pragma omp for 

//cout<<"=============================="<<endl;
        for (int i = 0; i < K; i++)
        {
	 //   #pragma omp simd
            for (int m = 0; m < E; ++m)
            {
                mean[i].elements[m] = tep[i].elements[m] / (count1[i]*100000);
            }
        }

	#pragma omp barrier
        //==========================================
        //根据新的聚类中心，重新进行聚类，原cluster函数
        //#pragma omp for private(minn,d),schedule(static, N/NUM_THREAD)
 	//#pragma omp single
	#pragma omp for private(minn,d),schedule(dynamic,1)
        for (int i = 0; i < N; ++i)
        {
            minn = (float)INT_MAX;
            for (int j = 0; j < K; ++j)
            {
                d = 0.0;
	//	#pragma omp simd
                for (int k = 0; k < E; k++)
                {
                    d += (point[i].elements[k] - mean[j].elements[k]) * (point[i].elements[k] - mean[j].elements[k]);
                }
                if (d < minn)
                {
                    minn = d;
                    center[i] = j;
                }
            }
            //printPointInfo(i);
        }
	#pragma omp barrier
        if(iter_num<MAX_ITER-1) continue;
        //float cnt = 0.0;//不能放在这里，要放在全局，private的变量要在全局定义and初始化？
       // for (int i = 0; i < K; ++i)
        {
            #pragma omp for schedule(dynamic, 1),reduction(+:global_sum)
            for (int j = 0; j < N; ++j)
            {

                //if (i == center[j])
                {
                    //cnt = (point[j].x - mean[i].x) * (point[j].x - mean[i].x) + (point[j].y - mean[i].y) * (point[j].y - mean[i].y);
         //           #pragma omp simd
			for (int k = 0; k < E; k++)
                    {
                        global_sum += (point[j].elements[k] - mean[center[j]].elements[k]) * (point[j].elements[k] - mean[center[j]].elements[k]);
                    }
                    
                }
            }
        }
    }
}
//========================================================================================================


void initPointSet()
{
	srand((unsigned int)time(NULL));
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < E; ++j)
		{
			point[i].elements[j] = rand() % 25;
		}
	}
}
bool checkFlag(int* flag, int j)//flag中没有j时返回真
{
	for (int i = 0; i < K; i++)
	{
		if (flag[i] == j)
		{
			return false;
		}
	}
	return true;
}
void initCenter()
{
	int i, j, n = 0;
	int flag[K];
	for (i = 0; i < K; i++)
	{
		flag[i] = -1;
	}
	srand((unsigned int)time(NULL));
	for (i = 0; i < K; ++i)
	{
		while (true)
		{
			j = rand() % N;
			if (checkFlag(flag, j))
			{
				mean[i] = point[j];
				center[j]=i;
				flag[i] = j;
				break;
			}

		}
		for (int e = 0; e < E; ++e)
		{
			mean[i].elements[e] = point[j].elements[e];
		}


	}
}

void kmeans_helper()
{
    struct timeval t1, t2;
     double total_time = 0.0;
	int n = 0;
	initCenter();
	cluster();
	n++;
    gettimeofday(&t1, NULL);
	while (n < MAX_ITER) { kmeans(); n++; }
getE();
    gettimeofday(&t2, NULL);
    total_time = (double)(t2.tv_sec - t1.tv_sec) * 1000.0 + (double)(t2.tv_usec - t1.tv_usec) / 1000.0;
    cout<<"total time of common kmeans alg:"<<total_time<<endl;
    cout<<"the Err is"<<getE()<<endl;
    /*
	for (int i = 0; i < N; ++i)
	{
		printPointInfo(i);
	}
	for (int j = 0; j < K; j++)
	{
		printCenterInfo(j);
	}
    */
}
void kmeans_openMP_helper()
{
	int n = 0;
	global_sum = 0;
	initCenter();
	cluster();
	n++;
	struct timeval t1, t2;
	double total_time = 0.0;
    gettimeofday(&t1, NULL);
    kmeans_openMP();
    gettimeofday(&t2, NULL);
     total_time = (double)(t2.tv_sec - t1.tv_sec) * 1000.0 + (double)(t2.tv_usec - t1.tv_usec) / 1000.0;
    cout<<"total time of paraller kmeans alg with openmp:"<<total_time<<endl;
    cout<<"the Err is"<<getE()<<endl;
    cout<<global_sum<<endl;
/*	
	for (int i = 0; i < N; ++i)
	{
		printPointInfo(i);
	}
	for (int j = 0; j < K; j++)
	{
		printCenterInfo(j);
	}
*/
	
}
int main()
{
	initPointSet();
	kmeans_helper();
	kmeans_openMP_helper();

	system("pause");
	return 0;
}
