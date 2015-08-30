// Distributed two-dimensional Discrete FFT transform
// YOUR NAME HERE
// ECE8893 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Transpose(Complex* src, int N, int M, Complex* dst){
  for(int n = 0; n<N*M; n++) {
    int i = n/N;
    int j = n%N;
    dst[n] = src[M*j + i];
  }
}

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  int N = w;
  for(int n=0; n< N; ++ n){
    Complex sum;
    for(int k=0; k < N ; ++k){
      sum = sum + Complex(cos(2*M_PI*n*k/N), -sin(2*M_PI*n*k/N)) * h[k];
    }
    H[n] = sum;
  }
}

void Transform2D(const char* inputFN)
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().
  int width, height;
  InputImage image(inputFN);  // Create the helper object for reading the image
  width = image.GetWidth();
  height = image.GetHeight();
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9
  //2 find the number of CPU
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int numRow ;
  if( rank == size-1) numRow = min(height- static_cast<int>(ceil(height/size))*(size-1), static_cast<int>(ceil(height/size)));
  else numRow = static_cast<int>(ceil(height/size)) ;
  int numElem = numRow * width;
  Complex* h = new Complex[ numElem];
  Complex* H = new Complex[ numElem];
  cout<<rank<<" " <<numElem<<endl;
  memcpy( h, image.GetImageData()+ numElem * rank, numElem *sizeof(Complex) );
  for( int i=0 ; i< numRow ; ++i) Transform1D(h+i*width, width, H+i*width) ;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Request request;
  MPI_Status status;
  MPI_Isend(H,numElem*sizeof(Complex),MPI_CHAR,0,0,MPI_COMM_WORLD, &request);
  cout<< "all 1d is done " <<endl;
  if(rank == 0){
    int rc;
    Complex* rh = new Complex[ width*height] ;
    for(int i=0 ;i < size; ++i)  {
      rc=MPI_Recv(rh+ numElem*i,numElem*sizeof(Complex),MPI_CHAR,i,0, MPI_COMM_WORLD,&status);
      if (rc != MPI_SUCCESS){
        cout << "Rank " << i
         << " send failed, rc " << rc << endl;
         MPI_Finalize();
         exit(1);
      }
    }
    image.SaveImageData("MyAfter1d.txt", rh , width, height);
    cout<<"One dimension transformation is saved"<<endl;
    Complex* transposed = new Complex[ width*height] ;
    Transpose(rh, width, height, transposed);
    memcpy(h,transposed,numElem*sizeof(Complex));
    cout<<"beginnign to send"<<endl;
    for(int i=1; i < size; ++i) MPI_Isend(transposed+i*numElem,numElem*sizeof(Complex),MPI_CHAR,i,0,MPI_COMM_WORLD,&request);
    delete[] rh;
    delete[] transposed;
    cout<<"sending to individual process of transformed array"<<endl;
  }
  // 1d transformation is completed, now continue to debug second part
    if(rank !=0) MPI_Recv(h,numElem*sizeof(Complex),MPI_CHAR,0,0,MPI_COMM_WORLD,&status);
    for( int i=0 ; i< numRow ; ++i) Transform1D(h+i*width, width, H+i*width) ;
    cout<<"ending second transform"<<endl;

    // if(rank != 0) MPI_Isend(H,numElem*sizeof(Complex),MPI_CHAR,0,0,MPI_COMM_WORLD,&request);
    // if(rank == 0){
    //   Complex* transposed = new Complex[ width*height] ;
    //   Complex* rh = new Complex[ width*height] ;
    //   for(int i=1 ;i < size; ++i) MPI_Recv(rh+ numElem*i,numElem*sizeof(Complex),MPI_CHAR,i,0, MPI_COMM_WORLD,&status);
    //   memcpy(rh,h,numElem*sizeof(Complex));
    //   Transpose(rh,width,height,transposed);
    //   image.SaveImageData("after2d_my_solution.txt", transposed , width, height);
    //   delete[] rh;
    //   delete[] transposed;
    // }
      MPI_Barrier(MPI_COMM_WORLD);
      Complex* transposed;
      Complex* rh ;
      cout<<"beginnning to gather"<<endl;
      if(rank==0){
        transposed = new Complex[ width*height];
        rh= new Complex[ width*height];
      }
      MPI_Gather(H,numElem*sizeof(Complex),MPI_CHAR,rh,numElem*sizeof(Complex),MPI_CHAR,0,MPI_COMM_WORLD);
      if(rank ==0){
        Transpose(rh,width,height,transposed);
        image.SaveImageData("MyAfter2d.txt", transposed , width, height);
        delete[] rh;
        delete[] transposed;
      }
      delete[] h;
      delete[] H;
      return ;
}


int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  int rc;
  rc = MPI_Init(&argc,&argv);
if (rc != MPI_SUCCESS) {
  printf ("Error starting MPI program. Terminating.\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
}
  Transform2D(fn.c_str()); // Perform the transform.
  // Finalize MPI here
    MPI_Finalize();
    return 0;

}
