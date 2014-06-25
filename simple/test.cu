#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>

#define TILE_WIDTH 512


struct vertex_buffer{
float x;
float y;
float z;
};

struct index_buffer{
int p1;
int p2;
int p3;
};

struct vertex_buffer *v;
struct index_buffer *p;
int vertices;
int faces;

void initialize(short int *q)
{
int i;
int j;

for(i=0;i<vertices;i++)
for(j=0;j<vertices;j++)
if(i==j)*(q+i*vertices+j)=1;
else
*(q+i*vertices+j)=0;
}

void create_adjacency_lists(short int *q)
{
int i;
for(i=0;i<faces;i++){
*(q+p[i].p1*vertices+p[i].p2)=1;
*(q+p[i].p2*vertices+p[i].p1)=1;
*(q+p[i].p1*vertices+p[i].p3)=1;
*(q+p[i].p3*vertices+p[i].p1)=1;
*(q+p[i].p2*vertices+p[i].p3)=1;
*(q+p[i].p3*vertices+p[i].p2)=1;
}
}

void __global__ laplacian (short int *p,int vertices,struct vertex_buffer *v,struct vertex_buffer *v1,short int *count,int i)
{
__shared__ short int lists[TILE_WIDTH];
__shared__ struct vertex_buffer temp[TILE_WIDTH];
int tx=threadIdx.x;
int bx=blockIdx.x;
int stride;
int column=bx*blockDim.x+tx;

	if(column < vertices){
	lists[tx]=*(p+0*vertices+column);
	temp[tx].x=lists[tx] * v[column].x;
	temp[tx].y=lists[tx] * v[column].y;
	temp[tx].z=lists[tx] * v[column].z;
	}
	else
	{
	lists[tx]=0;
	temp[tx].x=0;
	temp[tx].y=0;
	temp[tx].z=0;
	}

    for(stride=blockDim.x/2;stride>=1;stride >>= 1){
          __syncthreads();
          if(stride>tx){
            lists[tx]+=lists[tx+stride];
	    temp[tx].x+=temp[tx+stride].x;
	    temp[tx].y+=temp[tx+stride].y;	
	    temp[tx].z+=temp[tx+stride].z;
	  }
        }

if(tx==0){

*(count+0*((vertices-1)/TILE_WIDTH+1)+bx)=lists[tx];
v1[0*((vertices-1)/TILE_WIDTH+1)+bx].x=temp[tx].x;
v1[0*((vertices-1)/TILE_WIDTH+1)+bx].y=temp[tx].y;
v1[0*((vertices-1)/TILE_WIDTH+1)+bx].z=temp[tx].z;
}

}
  


int main()
{

FILE *fp=fopen("hbudda.off","r");

int edges;
char header[10];
int i,j;
int points;
short int *lists;
short int *d_lists;
struct vertex_buffer *d_v,*d_v1;
struct vertex_buffer *h_v1;
short int *d_count;
short int *h_count;

fscanf(fp,"%s",header);


if(!strcmp(header,"OFF\n")){
printf("Invalid Header");
exit(0);
}

fscanf(fp,"%d %d %d",&vertices,&faces,&edges);


v=(struct vertex_buffer*)malloc(sizeof(struct vertex_buffer)*vertices);
p=(struct index_buffer*)malloc(sizeof(struct index_buffer)*faces);

for(i=0;i<vertices;i++){
fscanf(fp,"%f %f %f",&v[i].x,&v[i].y,&v[i].z);
//printf("%f %f %f\n",v[i].x,v[i].y,v[i].z);
}

for(i=0;i<faces;i++)
{
fscanf(fp,"%d %d %d %d",&points,&p[i].p1,&p[i].p2,&p[i].p3);
//printf("%d %d %d %d\n",points,p[i].p1,p[i].p2,p[i].p3);
}

h_count=(short int*)malloc(sizeof(int)*((vertices-1)/TILE_WIDTH+1) * vertices);
cudaMalloc((void**)&d_count,sizeof(short int)*((vertices -1)/TILE_WIDTH + 1));	
//lists=(int *)malloc(sizeof(int)*vertices*vertices);
cudaHostAlloc((void**)&lists,sizeof(short int)*vertices*vertices,cudaHostAllocDefault);
initialize(lists);
create_adjacency_lists(lists);
cudaMalloc((void**)&d_lists,sizeof(short int)*vertices);
//cudaMemcpy(d_lists,lists,sizeof(int)*vertices*vertices,cudaMemcpyHostToDevice);

cudaMalloc((void**)&d_v,sizeof(struct vertex_buffer)*vertices);
cudaMalloc((void**)&d_v1,sizeof(struct vertex_buffer)*((vertices -1)/TILE_WIDTH + 1));
//h_v1=(struct vertex_buffer *)malloc(sizeof(struct vertex_buffer)*vertices* ((vertices -1)/TILE_WIDTH + 1));
cudaHostAlloc((void**)&h_v1,sizeof(struct vertex_buffer)*vertices*((vertices -1)/TILE_WIDTH + 1),cudaHostAllocDefault);
cudaMemcpy(d_v,v,sizeof(struct vertex_buffer)*vertices,cudaMemcpyHostToDevice);



dim3 dimGrid( (vertices-1) / TILE_WIDTH + 1 , 1,1);
dim3 dimBlock(TILE_WIDTH,1,1);

clock_t t_start;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
// Start record
cudaEventRecord(start, 0);
// Do something on GPU

for(i=0;i<vertices;i++){
cudaMemcpy(d_lists,lists+i*vertices,sizeof(short int)*vertices,cudaMemcpyHostToDevice);
laplacian<<<dimGrid,dimBlock>>>(d_lists,vertices,d_v,d_v1,d_count,i);
cudaDeviceSynchronize();
cudaMemcpy(h_count+i*((vertices-1)/TILE_WIDTH+1),d_count,sizeof(short int)*((vertices-1)/TILE_WIDTH+1),cudaMemcpyDeviceToHost);
cudaMemcpy(h_v1+i*((vertices-1)/TILE_WIDTH+1),d_v1,sizeof(struct vertex_buffer)*((vertices-1)/TILE_WIDTH+1), cudaMemcpyDeviceToHost);
}

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
// Clean up:
cudaEventDestroy(start);
cudaEventDestroy(stop);
clock_t t_stop;
printf("Time elapsed = %0.8f seconds\n",elapsedTime/1000);

//cudaMemcpy(h_count,d_count,sizeof(int)*((vertices-1)/TILE_WIDTH+1) * vertices, cudaMemcpyDeviceToHost);
//cudaMemcpy(h_v1,d_v1,sizeof(struct vertex_buffer)*((vertices-1)/TILE_WIDTH+1) * vertices, cudaMemcpyDeviceToHost);

t_start=clock();
int temp=(vertices-1)/TILE_WIDTH+1;
for(i=0;i<vertices;i++){

for(j=1;j<temp;j++){
*(h_count+i*temp)+=*(h_count+i*temp+j);
h_v1[i*temp].x+=h_v1[i*temp+j].x;
h_v1[i*temp].y+=h_v1[i*temp+j].y;
h_v1[i*temp].z+=h_v1[i*temp+j].z;
}
printf("%f %f %f\n",h_v1[i*temp].x/h_count[i*temp],h_v1[i*temp].y/h_count[i*temp],h_v1[i*temp].z/h_count[i*temp]);
//printf("count=%d\n",h_count[i*temp]);
}

t_stop=clock();

printf("Time elapsed = %0.8f seconds\n",(t_stop-t_start)/(float)CLOCKS_PER_SEC);


cudaFree(d_v);
cudaFree(d_lists);
cudaFree(d_v1);

free(v);
free(lists);

return 0;
}
