#include<stdio.h>

long int getimageinfo(FILE *fp,long int start, long int total_bytes)
{
	int j;
	long int i;
	unsigned char c;
	long int total=0;

	fseek(fp,start,SEEK_SET);


	for(j=0,i=start;i<(start+total_bytes);i++,j++)
	{
		fread(&c,sizeof(unsigned char),1,fp);
		total+=pow(256,j)*(int)c;
	}

return total;
}

void copy_info(FILE *fpi, FILE *fpo, long int total_bytes)
{
	int j;
	unsigned char c;

	fseek(fpi,0,SEEK_SET);

	for(j=0;j<total_bytes;j++)
	{
		fread(&c,sizeof(unsigned char),1,fpi);
		fwrite(&c,sizeof(unsigned char),1,fpo);
	}
}

void fetch_image_data(FILE *fpi, unsigned char *data, long int row, long int column)
{
	int i;
	int j;
	

	for(i=0;i<row;i++)
	for(j=0;j<column;j++)
	{
        fscanf(fpi,"%c",(data+i*column+j));
	}

}



void print_read_data(unsigned char *data,long int row, long int column)
{
	int i;
	int j;
	FILE *fp=fopen("matrix","w");


	for(i=0;i<row;i++){
	for(j=0;j<column;j++)
		fprintf(fp,"\n(%u,%u)=%u",i,j,*(data+i*column+j));
	
	}
fclose(fp);	
}
	
	
void copy_fetch_data(FILE *fpo,unsigned char *data,long int row, long int column)
{
	int i;
	int j;

	for(i=0;i<row;i++)
	for(j=0;j<column;j++)
		fwrite((data+i*column+j),sizeof(unsigned char),1,fpo);


}

