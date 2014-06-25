#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include "parse.cu"
#include <time.h>

#define TILE_WIDTH 16

__constant__ short int Gx[][3]={{-1,0,1},{-2,0,+2},{-1,0,+1}};
__constant__ short int Gy[][3]={{1,2,1},{0,0,0},{-1,-2,-1}};


__global__ void sobel_edge_detection(unsigned char *device_p, unsigned char *device_edge,int rows, int columns)
{
	int ty=threadIdx.y;
	int tx=threadIdx.x;
	int by=blockIdx.y;
	int bx=blockIdx.x;
	int row=by*TILE_WIDTH+ty;
	int column=bx*TILE_WIDTH+tx;
	int sumx;
	int sumy;
	int sum;
	int I;
	int J;


	if(row<rows && column<columns)
	{
		if(row==0 || row==rows-1|| column==0 || column==columns-1)
		sum=0;

		else{
		   sumx=0;
		   for(I=-1;I<=1;I++)
		   for(J=-1;J<=1;J++){
		   sumx+=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];
			}

		   sumy=0;
		   for(I=-1;I<=1;I++)
		   for(J=-1;J<=1;J++){
		   sumy+=(int)(*(device_p+(I+row)*columns+J+column))*(Gy[I+1][J+1]);
			}

		sum=abs(sumx)+abs(sumy);
	}

		if(sum>255)sum=255;
			
		*(device_edge+row*columns+column)=255-sum;
	}

}



int main(int argc, char **argv)
{
FILE *bmpinput;
FILE *bmpoutput;
unsigned long int num_rows;
unsigned long int num_columns;
unsigned long int num_colors;
unsigned char *host_p;
unsigned char *device_p;
unsigned char *host_edge;
unsigned char *device_edge;
cudaError_t err;
char header[3];
clock_t t_start;
clock_t t_end;


	if(argc!=3)
	{
		printf("<usuage> agruments mismatched\n");
		exit(0);
	}

	if((bmpinput=fopen(argv[1],"rb"))==NULL)
	{
		printf("could not open input bitmap file\n");
		exit(0);
	}

	if((bmpoutput=fopen(argv[2],"wb"))==NULL)
	{
		printf("could not open output bitmap file\n");
		exit(0);
	}


	//saving header information

	fscanf(bmpinput,"%s",header);
        fscanf(bmpinput,"%lu %lu",&num_columns, &num_rows);
        fscanf(bmpinput,"%lu",&num_colors);


        printf("num_columns:%lu\n",num_columns);
        printf("num_rows:%lu\n",num_rows);
        printf("num_colors:%lu\n",num_colors);

        fprintf(bmpoutput,"%s\n",header);
        fprintf(bmpoutput,"%lu %lu\n",num_columns,num_rows);
        fprintf(bmpoutput,"%lu\n",num_colors);





//	host_p=(unsigned char *)malloc(sizeof(unsigned char)*num_rows*num_columns);

	cudaHostAlloc((void**)&host_p,sizeof(unsigned char)*num_rows*num_columns,cudaHostAllocDefault);
	
		
	fetch_image_data(bmpinput,host_p,num_rows,num_columns);
	

	//print_read_data(p,num_rows,num_columns);

	//memory allocation for host to store the final result
	//host_edge=(unsigned char *)malloc(sizeof(unsigned char)*num_rows*num_columns);

	cudaHostAlloc((void**)&host_edge,sizeof(unsigned char)*num_rows*num_columns,cudaHostAllocDefault);



	//memory allocation for device pointer used by kernel

	err=cudaMalloc((void**)&device_p,sizeof(unsigned char)*num_rows*num_columns);

	if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,__LINE__);
           return 0;
      }
              
	cudaMalloc((void**)&device_edge,sizeof(unsigned char)*num_rows*num_columns);


        cudaMemcpy(device_p,host_p,sizeof(unsigned char)*num_rows*num_columns,cudaMemcpyHostToDevice);

	//grid and thread block allocation
       dim3 dimGrid( (num_columns-1) / TILE_WIDTH + 1 , (num_rows-1) / TILE_WIDTH + 1,1);
       dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);


	sobel_edge_detection<<<dimGrid,dimBlock>>>(device_p,device_edge,num_rows,num_columns);

        cudaThreadSynchronize();

	cudaMemcpy(host_edge,device_edge,sizeof(unsigned char)*num_rows*num_columns,cudaMemcpyDeviceToHost);

	//print_read_data(host_edge,num_rows,num_columns);

	copy_fetch_data(bmpoutput,host_edge,num_rows,num_columns);

	cudaFree(device_p);
	cudaFree(device_edge);

	cudaFreeHost(host_p);
	cudaFreeHost(host_edge);
	
	fclose(bmpinput);
	fclose(bmpoutput);

	return 0;
}

