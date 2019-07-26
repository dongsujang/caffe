#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
using namespace cv;
using namespace std;

void convolution(int rgb, int o_size_row, int o_size_col, int stride, int filter_size, int***pad_input,int **filter, int***output)
{
	int k,m,n,i,j,x,y;
	
	//convolution
	
	
	for(k=0;k<rgb;k++){
		for(m=0;m<o_size_row;m++){
			for(n=0;n<o_size_col;n++){
				for(i=0;i<filter_size;i++){
					for(j=0;j<filter_size;j++){
						output[k][m][n] = output[k][m][n] + pad_input[k][i+m*stride][j+n*stride]*filter[i][j];						
					}
				}
				if(output[k][m][n] > 255){
					output[k][m][n] = 255;
				}
				else if(output[k][m][n] < 0){
					output[k][m][n] = 0;
				}
			}
		}
	}

	
}


void pooling(int rgb, int p_out_size_row, int p_out_size_col ,int filter_size, int stride, int ***pool_output, int ***input)
{
	int i,j,m,n,k,x,y;
	int max;

	for(k=0;k<rgb;k++){
		for(m=0;m<p_out_size_row;m++){
			for(n=0;n<p_out_size_col;n++){
				max = 0;
				for(i=0;i<filter_size;i++){
					for(j=0;j<filter_size;j++){
						if(max < input[k][i+m*stride][j+n*stride]){
							max = input[k][i+m*stride][j+n*stride];
						}
					}
				}
				pool_output[k][m][n] = max;
			}
		}
	}	
}

int main()
{
	int i,j,m,n,k,x,y;
	int ***output;
	int ***pad_input;
	int pad = 1;
	int stride = 0;
	int input_size=0;
	int filter_size=0;
	int ***input;
	int **filter;
	int rgb = 3;
	int max;
	int ***pool_output;
	int p_out_size_row = 0;
	int p_out_size_col = 0;
	int mode;

	int o_size_row = 0;
	int o_size_col = 0;


	//이미지 불러오기
	Mat image;
	image = imread("test2.jpg", IMREAD_COLOR);
	
	if(image.empty())
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	 // 입력 동적할당(이미지에따른)
	input = (int ***)malloc(rgb*sizeof(int **));
	
	for(i=0; i<rgb;i++){
		*(input+i) = (int **)malloc(image.rows*sizeof(int *));
		for(j=0;j < image.rows;j++){
			*(*(input+i)+j) = (int *)malloc(image.cols*sizeof(int *));
		}
	}
	//
	for(k=0;k<rgb;k++){
		for(x=0;x<image.rows;x++){
			for(y=0;y<image.cols;y++){
				input[k][x][y] = image.at<Vec3b>(x,y)[k];
				
			}
		}
	}

	//padding 입력값
	printf("padding : " );
	scanf("%d", &pad);

	//stride 입력값
	printf("stride : " );
	scanf("%d", &stride);


	//패딩 처리 행렬
	pad_input = (int ***)malloc(rgb*sizeof(int **));

	for(i=0;i<rgb;i++){
		*(pad_input+i) = (int **)malloc((image.rows+pad*2)*sizeof(int *));
		for(j=0;j<(image.rows+pad*2);j++){
			*(*(pad_input+i)+j) = (int *)malloc((image.cols+pad*2)*sizeof(int));
		}
	}
	//패딩행렬 초기화
	for(m=0;m<rgb;m++){
		for(i=0;i<(image.rows+pad*2);i++){
			for(j=0;j<(image.cols+pad*2);j++){
				pad_input[m][i][j] = 0;	
			}
		}
	}

	// pad_input 에 값 전달
	for(m=0;m<rgb;m++){
		for(i=0; i<image.rows;i++){
			for(j=0;j<image.cols;j++){
				pad_input[m][i+pad*1][j+pad*1] = input [m][i][j];
			}
		}
	}

	//filter size 입력
	printf("filter size : " );
	scanf("%d", &filter_size);
	//filter 할당
	filter = (int **)malloc(filter_size*sizeof(int *));
	for(i=0; i<filter_size;i++){
		*(filter+i) = (int *)malloc(filter_size*sizeof(int));
	}

	// filter 입력 받기
	for(i=0;i<filter_size;i++){
		for(j=0;j<filter_size;j++){		
			scanf("%d", &filter[i][j]);
		}	
	}

	
	o_size_row = ((image.rows-filter_size+2*pad)/stride)+1; // output_size
	o_size_col = ((image.cols-filter_size+2*pad)/stride)+1;	

	// 출력 동적할당으로 생성
	//output 생성

	output = (int ***)malloc(rgb*sizeof(int **));
	
	for(i=0; i<rgb;i++){
	
		*(output+i) = (int **)malloc(o_size_row*sizeof(int *));
		for(j=0;j < o_size_row ;j++){
		
			*(*(output+i)+j) = (int *)malloc(o_size_col*sizeof(int *));
		}
	
	}

	

	// output 초기화
	for(m=0;m<rgb;m++){
		for(i=0;i<o_size_row;i++){
			for(j=0;j<o_size_col;j++){
				output[m][i][j] = 0;
			}		
		}
	}

	p_out_size_row = ((image.rows-filter_size)/stride)+1;
	p_out_size_col = ((image.cols-filter_size)/stride)+1;

	pool_output = (int ***)malloc(rgb*sizeof(int **));

	for(i=0; i<rgb;i++){
		*(pool_output+i) = (int **)malloc(p_out_size_row*sizeof(int *));
		for(j=0;j < p_out_size_row;j++){
			*(*(pool_output+i)+j) = (int *)malloc(p_out_size_col*sizeof(int *));
		}
	}



	printf(" 1: conv , 2: pooling  :  ");
	scanf("%d",&mode);

	if(mode==1)
	{

	convolution(rgb,o_size_row,o_size_col,stride,filter_size,pad_input,filter,output);
	
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", image);

	Mat result(o_size_row,o_size_col,image.type());

	for(k=0;k<rgb;k++){
		for(x=0;x<o_size_row;x++){
			for(y=0;y<o_size_col;y++){
				result.at<Vec3b>(x,y)[k] = output[k][x][y];
			}
		}
	}

	imwrite("edge_3.jpg", result);

	namedWindow("Original2", WINDOW_AUTOSIZE);
	imshow("Original2", result);

	waitKey(0);

	}
	else{
	
	pooling(rgb,p_out_size_row,p_out_size_col,filter_size,stride,pool_output,input);

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", image);

	Mat result(p_out_size_row,p_out_size_col,image.type());

	for(k=0;k<rgb;k++){
		for(x=0;x<p_out_size_row;x++){
			for(y=0;y<p_out_size_col;y++){
				result.at<Vec3b>(x,y)[k] = pool_output[k][x][y];
			}
		}
	}

	imwrite("result2.jpg", result);

	namedWindow("Original2", WINDOW_AUTOSIZE);
	imshow("Original2", result);

	waitKey(0);
	}




	




	
	for(i=0;i<filter_size;i++){
		free(*(filter+i));
	}
	free(filter);
	
	for(i=0;i<rgb;i++){
		for(j=0;j<o_size_row;j++){
			free(*(*(output+i)+j));
		}
		free(*(output+i));
	}
	free(output);

	for(i=0;i<rgb;i++){
		for(j=0;j<input_size;j++){
			free(*(*(input+i)+j));
		}
		free(*(input+i));
	}
	free(input);
	
	for(i=0;i<rgb;i++){
		for(j=0;j<(input_size+pad*2);j++){
			free(*(*(pad_input+i)+j));
		}
		free(*(pad_input+i));
	}
	free(pad_input);

	return 0;
}



















/*
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
using namespace cv;
using namespace std;

void convolution(int rgb, int o_size_row, int o_size_col, int stride, int filter_size, int***pad_input, int ***filter, int***output)
{
	int k,m,n,i,j,x,y,filter_rgb;
	
	//convolution
	
	
	for(k=0;k<rgb;k++){
		for(filter_rgb=0;filter_rgb<rgb;filter_rgb++){
			for(m=0;m<o_size_row;m++){
				for(n=0;n<o_size_col;n++){
					for(i=0;i<filter_size;i++){
						for(j=0;j<filter_size;j++){
							output[k][m][n] = output[k][m][n] + pad_input[k][i+m*stride][j+n*stride]*filter[filter_rgb][i][j];				
						}
					}
					if(output[k][m][n] > 255){
						output[k][m][n] = 255;
					}
					else if(output[k][m][n] < 0){
						output[k][m][n] = 0;
					}
				}
			}
		}
	}
}

void pooling(int rgb, int p_out_size_row, int p_out_size_col ,int filter_size, int stride, int ***pool_output, int ***input)
{
	int i,j,m,n,k,x,y;
	int max;

	for(k=0;k<rgb;k++){
		for(m=0;m<p_out_size_row;m++){
			for(n=0;n<p_out_size_col;n++){
				max = 0;
				for(i=0;i<filter_size;i++){
					for(j=0;j<filter_size;j++){
						if(max < input[k][i+m*stride][j+n*stride]){
							max = input[k][i+m*stride][j+n*stride];
						}
					}
				}
				pool_output[k][m][n] = max;
			}
		}
	}	
}

void activation(int rgb, int***sigma, int o_size_row, int o_size_col, int ***f_sigma)
{
	int i,j,k;

	for(k=0;k<rgb;k++){
		for(i=0;i<o_size_row;i++){
			for(j=0;j<o_size_col;j++){
				f_sigma[k][i][j] = sigma[k][i][j];
			}
		}
	}
}

int main()
{
	int i,j,m,n,k,x,y;
	int ***output;
	int ***pad_input;
	int pad = 1;
	int stride = 0;
	int input_size=0;
	int filter_size=0;
	int ***input;
	int ***filter;
	int rgb = 3;
	int max;
	int ***pool_output;
	int p_out_size_row = 0;
	int p_out_size_col = 0;
	int mode;
	int ***f_sigma;

	int o_size_row = 0;
	int o_size_col = 0;


	//이미지 불러오기
	Mat image;
	image = imread("test.jpg", IMREAD_COLOR);
	
	if(image.empty())
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	 // 입력 동적할당(이미지에따른)
	input = (int ***)malloc(rgb*sizeof(int **));
	
	for(i=0; i<rgb;i++){
		*(input+i) = (int **)malloc(image.rows*sizeof(int *));
		for(j=0;j < image.rows;j++){
			*(*(input+i)+j) = (int *)malloc(image.cols*sizeof(int));
		}
	}
	//
	for(k=0;k<rgb;k++){
		for(x=0;x<image.rows;x++){
			for(y=0;y<image.cols;y++){
				input[k][x][y] = image.at<Vec3b>(x,y)[k];
				
			}
		}
	}

	//padding 입력값
	printf("padding : " );
	scanf("%d", &pad);

	//stride 입력값
	printf("stride : " );
	scanf("%d", &stride);


	//패딩 처리 행렬
	pad_input = (int ***)malloc(rgb*sizeof(int **));

	for(i=0;i<rgb;i++){
		*(pad_input+i) = (int **)malloc((image.rows+pad*2)*sizeof(int *));
		for(j=0;j<(image.rows+pad*2);j++){
			*(*(pad_input+i)+j) = (int *)malloc((image.cols+pad*2)*sizeof(int));
		}
	}
	//패딩행렬 초기화
	for(m=0;m<rgb;m++){
		for(i=0;i<(image.rows+pad*2);i++){
			for(j=0;j<(image.cols+pad*2);j++){
				pad_input[m][i][j] = 0;	
			}
		}
	}

	// pad_input 에 값 전달
	for(m=0;m<rgb;m++){
		for(i=0; i<image.rows;i++){
			for(j=0;j<image.cols;j++){
				pad_input[m][i+pad*1][j+pad*1] = input [m][i][j];
			}
		}
	}

	//filter size 입력
	printf("filter size : " );
	
	scanf("%d", &filter_size);

	//filter 할당
	filter = (int ***)malloc(rgb*sizeof(int **));
	
	for(i=0; i<rgb;i++){
	
		*(filter+i) = (int **)malloc(filter_size*sizeof(int *));
		for(j=0;j < filter_size ;j++){
		
			*(*(filter+i)+j) = (int *)malloc(filter_size*sizeof(int));
		}
	
	}
	
	// filter 입력 받기
	for(m=0;m<rgb;m++){
		for(i=0;i<filter_size;i++){
			for(j=0;j<filter_size;j++){		
				scanf("%d", &filter[m][i][j]);
			}	
		}
	}

	
	o_size_row = ((image.rows-filter_size+2*pad)/stride)+1; // output_size
	o_size_col = ((image.cols-filter_size+2*pad)/stride)+1;	

	// 출력 동적할당으로 생성
	//output 생성

	output = (int ***)malloc(rgb*sizeof(int **));
	
	for(i=0; i<rgb;i++){
	
		*(output+i) = (int **)malloc(o_size_row*sizeof(int *));
		for(j=0;j < o_size_row ;j++){
		
			*(*(output+i)+j) = (int *)malloc(o_size_col*sizeof(int));
		}
	
	}

	

	// output 초기화
	for(m=0;m<rgb;m++){
		for(i=0;i<o_size_row;i++){
			for(j=0;j<o_size_col;j++){
				output[m][i][j] = 0;
			}		
		}
	}

	p_out_size_row = ((image.rows-filter_size)/stride)+1;
	p_out_size_col = ((image.cols-filter_size)/stride)+1;

	pool_output = (int ***)malloc(rgb*sizeof(int **));

	for(i=0; i<rgb;i++){
		*(pool_output+i) = (int **)malloc(p_out_size_row*sizeof(int *));
		for(j=0;j < p_out_size_row;j++){
			*(*(pool_output+i)+j) = (int *)malloc(p_out_size_col*sizeof(int));
		}
	}

	//activation sigma
	f_sigma = (int ***)malloc(rgb*sizeof(int **));
	
	for(i=0; i<rgb;i++){
	
		*(f_sigma+i) = (int **)malloc(o_size_row*sizeof(int *));
		for(j=0;j < o_size_row ;j++){
		
			*(*(f_sigma+i)+j) = (int *)malloc(o_size_col*sizeof(int));
		}
	
	}

	// sigma 초기화
	for(m=0;m<rgb;m++){
		for(i=0;i<o_size_row;i++){
			for(j=0;j<o_size_col;j++){
				f_sigma[m][i][j] = 0;
			}		
		}
	}

	printf(" 1: conv , 2: pooling  :  ");
	scanf("%d",&mode);

	if(mode==1)
	{

	convolution(rgb,o_size_row,o_size_col,stride,filter_size,pad_input,filter,output);
	
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", image);

	Mat result(o_size_row,o_size_col,image.type());

	for(k=0;k<rgb;k++){
		for(x=0;x<o_size_row;x++){
			for(y=0;y<o_size_col;y++){
				result.at<Vec3b>(x,y)[k] = output[k][x][y];
			}
		}
	}

	imwrite("edge_3.jpg", result);

	namedWindow("Original2", WINDOW_AUTOSIZE);
	imshow("Original2", result);


	waitKey(0);

	}
	else{
	
	pooling(rgb,p_out_size_row,p_out_size_col,filter_size,stride,pool_output,input);

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", image);

	Mat result(p_out_size_row,p_out_size_col,image.type());

	for(k=0;k<rgb;k++){
		for(x=0;x<p_out_size_row;x++){
			for(y=0;y<p_out_size_col;y++){
				result.at<Vec3b>(x,y)[k] = pool_output[k][x][y];
			}
		}
	}

	imwrite("result2.jpg", result);

	namedWindow("Original2", WINDOW_AUTOSIZE);
	imshow("Original2", result);

	waitKey(0);
	}

	activation(rgb, output, o_size_row, o_size_col, f_sigma);

	




	
	for(i=0;i<filter_size;i++){
		free(*(filter+i));
	}
	free(filter);
	
	for(i=0;i<rgb;i++){
		for(j=0;j<o_size_row;j++){
			free(*(*(output+i)+j));
		}
		free(*(output+i));
	}
	free(output);

	for(i=0;i<rgb;i++){
		for(j=0;j<input_size;j++){
			free(*(*(input+i)+j));
		}
		free(*(input+i));
	}
	free(input);
	
	for(i=0;i<rgb;i++){
		for(j=0;j<(input_size+pad*2);j++){
			free(*(*(pad_input+i)+j));
		}
		free(*(pad_input+i));
	}
	free(pad_input);

	for(i=0;i<rgb;i++){
		for(j=0;j<o_size_row;j++){
			free(*(*(f_sigma+i)+j));
		}
		free(*(f_sigma+i));
	}
	free(f_sigma);

	return 0;
}



















*/