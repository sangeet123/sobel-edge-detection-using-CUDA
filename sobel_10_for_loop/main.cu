#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include "parse.cu"
#include <sys/time.h>

#define TILE_WIDTH 16
/* copied from mpbench */
#define TIMER_CLEAR     (tv1.tv_sec = tv1.tv_usec = tv2.tv_sec = tv2.tv_usec = 0)
#define TIMER_START     gettimeofday(&tv1, (struct timezone*)0)
#define TIMER_ELAPSED   ((tv2.tv_usec-tv1.tv_usec)+((tv2.tv_sec-tv1.tv_sec)*1000000))
#define TIMER_STOP      gettimeofday(&tv2, (struct timezone*)0)
struct timeval tv1,tv2;

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
	int sumx[9];
	int sumy[9];
	int sum;
	int tempx=0;
	int tempy=0;
	int I;
	int J;


	if(row<rows && column<columns)
	{
		if(row==0 || row==rows-1|| column==0 || column==columns-1)
		sum=0;

		else{
		   I=-1;
		   J=-1;
		  sumx[0]=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];	
		  sumy[0]=(int)(*(device_p+(I+row)*columns+J+column))*Gy[I+1][J+1];
 
             
                   J=0;
                  sumx[1]=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];
                  sumy[1]=(int)(*(device_p+(I+row)*columns+J+column))*Gy[I+1][J+1];

		  
                   J=1;
                  sumx[2]=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];
                  sumy[2]=(int)(*(device_p+(I+row)*columns+J+column))*Gy[I+1][J+1];

                   I=0;
                   J=-1;
                  sumx[3]=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];
                  sumy[3]=(int)(*(device_p+(I+row)*columns+J+column))*Gy[I+1][J+1];

                  
                   J=0;
                  sumx[4]=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];
                  sumy[4]=(int)(*(device_p+(I+row)*columns+J+column))*Gy[I+1][J+1];

                   
                   J=1;
                  sumx[5]=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];
                  sumy[5]=(int)(*(device_p+(I+row)*columns+J+column))*Gy[I+1][J+1];

                   I=1;
                   J=-1;
                  sumx[6]=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];
                  sumy[6]=(int)(*(device_p+(I+row)*columns+J+column))*Gy[I+1][J+1];

        
                   J=0;
                  sumx[7]=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];
                  sumy[7]=(int)(*(device_p+(I+row)*columns+J+column))*Gy[I+1][J+1];

         
                   J=1;
                  sumx[8]=(int)(*(device_p+(I+row)*columns+J+column))*Gx[I+1][J+1];
                  sumy[8]=(int)(*(device_p+(I+row)*columns+J+column))*Gy[I+1][J+1];		  


		for(I=0;I<9;I++){
		tempx+=sumx[I];
		tempy+=sumy[I];
		}

		sum=abs(tempx)+abs(tempy);
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





	host_p=(unsigned char *)malloc(sizeof(unsigned char)*num_rows*num_columns);

//	cudaHostAlloc((void**)&host_p,sizeof(unsigned char)*num_rows*num_columns,cudaHostAllocDefault);
	
		
	fetch_image_data(bmpinput,host_p,num_rows,num_columns);
	

	//print_read_data(p,num_rows,num_columns);

	//memory allocation for host to store the final result
	host_edge=(unsigned char *)malloc(sizeof(unsigned char)*num_rows*num_columns);

//	cudaHostAlloc((void**)&host_edge,sizeof(unsigned char)*num_rows*num_columns,cudaHostAllocDefault);



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



	//Launching kernel

//	TIMER_CLEAR;
//        TIMER_START;

	t_start=clock();


	sobel_edge_detection<<<dimGrid,dimBlock>>>(device_p,device_edge,num_rows,num_columns);

        cudaDeviceSynchronize();

	 t_end=clock();


//	TIMER_STOP;
//	printf("Time elapsed = %0.8f seconds\n",TIMER_ELAPSED/1000000.0);

	printf("Time elapsed = %0.8f seconds\n",(t_end-t_start)/(float)CLOCKS_PER_SEC);

	cudaMemcpy(host_edge,device_edge,sizeof(unsigned char)*num_rows*num_columns,cudaMemcpyDeviceToHost);

	//print_read_data(host_edge,num_rows,num_columns);

	copy_fetch_data(bmpoutput,host_edge,num_rows,num_columns);

	cudaFree(device_p);
	cudaFree(device_edge);

	free(host_p);
	free(host_edge);
	
	fclose(bmpinput);
	fclose(bmpoutput);

	return 0;
}

