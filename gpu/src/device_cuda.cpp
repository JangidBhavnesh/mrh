/* -*- c++ -*- */

#if defined(_GPU_CUDA)

#include "device.h"

#include <stdio.h>

#define _DOT_BLOCK_SIZE 32
#define _TRANSPOSE_BLOCK_SIZE 32
#define _UNPACK_BLOCK_SIZE 32

/* ---------------------------------------------------------------------- */

void Device::init_get_jk(py::array_t<double> _eri1, py::array_t<double> _dmtril, int _blksize, int _nset, int _nao, int _naux, int count)
{
  //  printf("Inside init_get_jk()\n");
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  blksize = _blksize;
  nset = _nset;
  nao = _nao;
  naux = _naux;

  nao_pair = nao * (nao+1) / 2;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (232, 351)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, 351)

  //  double * eri1 = static_cast<double*>(info_eri1.ptr);
  //  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  
  int _size_vj = nset * nao_pair;
  if(_size_vj > size_vj) {
    size_vj = _size_vj;
    //if(vj) pm->dev_free_host(vj);
    //vj = (double *) pm->dev_malloc_host(size_vj * sizeof(double));
    if(d_vj) pm->dev_free(d_vj);
    d_vj = (double *) pm->dev_malloc(size_vj * sizeof(double));
  }
  //for(int i=0; i<_size_vj; ++i) vj[i] = 0.0;

  int _size_vk = nset * nao * nao;
  if(_size_vk > size_vk) {
    size_vk = _size_vk;
    //    if(_vktmp) pm->dev_free_host(_vktmp);
    //    _vktmp = (double *) pm->dev_malloc_host(size_vk*sizeof(double));

#ifdef _CUDA_NVTX
    nvtxRangePushA("Realloc");
#endif
    
    if(d_vkk) pm->dev_free(d_vkk);
    d_vkk = (double *) pm->dev_malloc(size_vk * sizeof(double));

#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
  }
  //  for(int i=0; i<_size_vk; ++i) _vktmp[i] = 0.0;

  int _size_buf = blksize * nao * nao;
  if(_size_buf > size_buf) {
    size_buf = _size_buf;
    if(buf_tmp) pm->dev_free_host(buf_tmp);
    if(buf3) pm->dev_free_host(buf3);
    if(buf4) pm->dev_free_host(buf4);
    
    buf_tmp = (double*) pm->dev_malloc_host(2*size_buf*sizeof(double));
    buf3 = (double *) pm->dev_malloc_host(size_buf*sizeof(double)); // (nao, blksize*nao)
    buf4 = (double *) pm->dev_malloc_host(size_buf*sizeof(double)); // (blksize*nao, nao)

#ifdef _CUDA_NVTX
    nvtxRangePushA("Realloc");
#endif

    if(d_buf1) pm->dev_free(d_buf1);
    if(d_buf2) pm->dev_free(d_buf2);
    if(d_buf3) pm->dev_free(d_buf3);
    
    d_buf1 = (double *) pm->dev_malloc(size_buf * sizeof(double));
    d_buf2 = (double *) pm->dev_malloc(size_buf * sizeof(double));
    d_buf3 = (double *) pm->dev_malloc(size_buf * sizeof(double));

#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
  }

  int _size_fdrv = nao * nao * num_threads;
  if(_size_fdrv > size_fdrv) {
    size_fdrv = _size_fdrv;
    if(buf_fdrv) pm->dev_free_host(buf_fdrv);
    buf_fdrv = (double *) pm->dev_malloc_host(size_fdrv*sizeof(double));
  }

  int _size_dms = nao * nao;
  if(_size_dms > size_dms) {
    size_dms = _size_dms;
    if(d_dms) pm->dev_free(d_dms);
    d_dms = (double *) pm->dev_malloc(size_dms * sizeof(double));
  }

  int _size_eri1 = naux * nao_pair;
  if(_size_eri1 > size_eri1) {
    size_eri1 = _size_eri1;
    if(d_eri1) pm->dev_free(d_eri1);
    d_eri1 = (double *) pm->dev_malloc(size_eri1 * sizeof(double));
  }
  
  int _size_tril_map = nao * nao;
  //  if(_size_tril_map > size_tril_map) {
  if(_size_tril_map != size_tril_map) {
    // nao can change between calls, so mapping needs to be updated...
    // I think there are only two mappings needed
    
    size_tril_map = _size_tril_map;
    if(tril_map) pm->dev_free_host(tril_map);
    if(d_tril_map) pm->dev_free(d_tril_map);
    tril_map = (int *) pm->dev_malloc_host(size_tril_map * sizeof(int));
    d_tril_map = (int *) pm->dev_malloc(size_tril_map * sizeof(int));

    // optimize map later...

    int _i, _j, _ij;
    for(_ij = 0, _i = 0; _i < nao; _i++)
      for(_j = 0; _j<=_i; _j++, _ij++) {
    	tril_map[_i*nao + _j] = _ij;
    	tril_map[_i + nao*_j] = _ij;
      }
    
    pm->dev_push_async(d_tril_map, tril_map, size_tril_map*sizeof(int), stream);
  }
  
  // Create cuda stream
  
  if(stream == nullptr) {
    pm->dev_stream_create(stream);
  }
  
  // Create blas handle

  if(handle == nullptr) {
#ifdef _CUDA_NVTX
    nvtxRangePushA("Create handle");
#endif
    cublasCreate(&handle);
    _CUDA_CHECK_ERRORS();
    cublasSetStream(handle, stream);
    _CUDA_CHECK_ERRORS();
#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
  }
  
#ifdef _SIMPLE_TIMER
  t_array_jk[0] += omp_get_wtime() - t0;
#endif
  //  printf("Leaving init_get_jk()\n");
}

/* ---------------------------------------------------------------------- */

void Device::pull_get_jk(py::array_t<double> _vj, py::array_t<double> _vk)
{
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  
  double * vj = static_cast<double*>(info_vj.ptr);
  double * vk = static_cast<double*>(info_vk.ptr);
 
  pm->dev_pull(d_vkk, vk, nset * nao * nao * sizeof(double));
  pm->dev_pull(d_vj, vj, nset * nao_pair * sizeof(double));
}

/* ---------------------------------------------------------------------- */

__global__ void _getjk_vj(double * vj, double * rho, double * eri1, int nset, int nao_pair, int naux, int count)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= nset) return;
  if(j >= nao_pair) return;

  double val = 0.0;
  for(int k=0; k<naux; ++k) val += rho[i * naux + k] * eri1[k * nao_pair + j];
  
  if(count == 0) vj[i * nao_pair + j] = val;
  else vj[i * nao_pair + j] += val;
}

/* ---------------------------------------------------------------------- */

__global__ void _getjk_unpack_buf2(double * buf2, double * eri1, int * map, int naux, int nao, int nao_pair)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= naux) return;
  if(j >= nao*nao) return;

  double * buf = &(buf2[i * nao * nao]);
  double * tril = &(eri1[i * nao_pair]);

  buf[j] = tril[ map[j] ];
  //  for(int j=0; j<nao*nao; ++j) buf[j] = tril[ map[j] ];
}

/* ---------------------------------------------------------------------- */

__global__ void _getjk_transpose_buf1_buf3(double * buf3, double * buf1, int naux, int nao)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i >= naux) return;
  if(j >= nao) return;

#if 1

  for(int k=0; k<nao; ++k) buf3[k * (naux*nao) + (i*nao+j)] = buf1[(i * nao + j) * nao + k];
  
#else
  DevArray3D da_buf1 = DevArray3D(buf1, naux, nao, nao);
  DevArray2D da_buf3 = DevArray2D(buf3, nao, naux * nao); // python swapped 1st two dimensions?
  
  //  for(int i=0; i<naux; ++i) {
  for(int j=0; j<nao; ++j)
    for(int k=0; k<nao; ++k) da_buf3(k,i*nao+j) = da_buf1(i,j,k);
    //  }
#endif
}

/* ---------------------------------------------------------------------- */

void Device::get_jk(int naux,
		    py::array_t<double> _eri1, py::array_t<double> _dmtril, py::list & _dms_list,
		    py::array_t<double> _vj, py::array_t<double> _vk,
		    int count)
{
  //  printf("Inside get_jk()\n");
#ifdef _SIMPLE_TIMER
  double t0 = omp_get_wtime();
#endif

  const int with_j = true;
  
  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair)
  py::buffer_info info_dmtril = _dmtril.request(); // 2D array (nset, nao_pair)
  py::buffer_info info_vj = _vj.request(); // 2D array (nset, nao_pair)

  double * eri1 = static_cast<double*>(info_eri1.ptr);
  double * dmtril = static_cast<double*>(info_dmtril.ptr);
  double * vj = static_cast<double*>(info_vj.ptr);
  
  int nao_pair = nao * (nao+1) / 2;
  
  pm->dev_push_async(d_eri1, eri1, naux * nao_pair * sizeof(double), stream);
  
  int _size_rho = info_dmtril.shape[0] * info_eri1.shape[0];
  if(_size_rho > size_rho) {
    size_rho = _size_rho;
    if(rho) pm->dev_free_host(rho);
    if(d_rho) pm->dev_free(d_rho);
    rho = (double *) pm->dev_malloc_host(size_rho * sizeof(double));
    d_rho = (double *) pm->dev_malloc(size_rho * sizeof(double));
  }

#if 0
  py::buffer_info info_vk = _vk.request(); // 3D array (nset, nao, nao)
  //  double * vk = static_cast<double*>(info_vk.ptr);
  printf("LIBGPU:: blksize= %i  naux= %i  nao= %i  nset= %i\n",blksize,naux,nao,nset);
  printf("LIBGPU::shape: dmtril= (%i,%i)  eri1= (%i,%i)  rho= (%i, %i)   vj= (%i,%i)  vk= (%i,%i,%i)\n",
  	 info_dmtril.shape[0], info_dmtril.shape[1],
  	 info_eri1.shape[0], info_eri1.shape[1],
  	 info_dmtril.shape[0], info_eri1.shape[0],
  	 info_dmtril.shape[0], info_eri1.shape[1],
  	 info_vk.shape[0],info_vk.shape[1],info_vk.shape[2]);
#endif
  
#ifdef _SIMPLE_TIMER
  t_array_jk[1] += omp_get_wtime() - t0;
#endif
  
  if(with_j) {

#ifdef _SIMPLE_TIMER
    double t0 = omp_get_wtime();
#endif

    DevArray2D da_rho = DevArray2D(rho, nset, naux);
    DevArray2D da_dmtril = DevArray2D(dmtril, nset, nao_pair);
    DevArray2D da_eri1 = DevArray2D(eri1, naux, nao_pair);
    
    // rho = numpy.einsum('ix,px->ip', dmtril, eri1)

#pragma omp parallel for collapse(2)
    for(int i=0; i<nset; ++i)
      for(int j=0; j<naux; ++j) {
	double val = 0.0;
	for(int k=0; k<nao_pair; ++k) val += da_dmtril(i,k) * da_eri1(j,k);
	da_rho(i,j) = val;
      }
    
#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
#endif
    
    DevArray2D da_vj = DevArray2D(vj, nset, nao_pair);
    
    // vj += numpy.einsum('ip,px->ix', rho, eri1)

#if 1
 {
   pm->dev_push_async(d_rho, rho, nset * naux * sizeof(double), stream);

   dim3 grid_size((nset + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE, (nao_pair + (_DOT_BLOCK_SIZE - 1)) / _DOT_BLOCK_SIZE, 1);
   dim3 block_size(_DOT_BLOCK_SIZE, _DOT_BLOCK_SIZE, 1);

   _getjk_vj<<<grid_size, block_size, 0, stream>>>(d_vj, d_rho, d_eri1, nset, nao_pair, naux, count);
  }
#else

#pragma omp parallel for collapse(2)
    for(int i=0; i<nset; ++i)
      for(int j=0; j<nao_pair; ++j) {

	double val = 0.0;
	for(int k=0; k<naux; ++k) val += da_rho(i,k) * da_eri1(k,j);
	da_vj(i,j) += val;
      }
#endif

#ifdef _SIMPLE_TIMER
    double t2 = omp_get_wtime();
    t_array_jk[2] += t1 - t0;
    t_array_jk[3] += t2 - t1;
#endif
  }
 
  double * buf1 = buf_tmp;
  double * buf2 = &(buf_tmp[blksize * nao * nao]);
    
  DevArray3D da_buf1 = DevArray3D(buf1, naux, nao, nao);
  DevArray2D da_buf2 = DevArray2D(buf2, blksize * nao, nao);
  DevArray2D da_buf3 = DevArray2D(buf3, nao, naux * nao); // python swapped 1st two dimensions?
  
#ifdef _SIMPLE_TIMER
  double t2 = omp_get_wtime();
#endif
    
  // buf2 = lib.unpack_tril(eri1, out=buf[1])

#if 1
  {
    //    pm->dev_push_async(d_eri1, eri1, naux * nao_pair * sizeof(double), stream);
    
    dim3 grid_size((naux + (_UNPACK_BLOCK_SIZE - 1)) / _UNPACK_BLOCK_SIZE, (nao*nao + (_UNPACK_BLOCK_SIZE - 1)) / _UNPACK_BLOCK_SIZE, 1);
    dim3 block_size(_UNPACK_BLOCK_SIZE, _UNPACK_BLOCK_SIZE, 1);
    
    _getjk_unpack_buf2<<<grid_size, block_size, 0, stream>>>(d_buf2, d_eri1, d_tril_map, naux, nao, nao_pair);
  }
#else
  
#pragma omp parallel for
    for(int i=0; i<naux; ++i) {
      double * buf = &(buf2[i * nao * nao]);
      double * tril = &(eri1[i * nao_pair]);
      for(int j=0; j<nao*nao; ++j) buf[j] = tril[ tril_map[j] ];
    }
    
    pm->dev_push_async(d_buf2, buf2, blksize * nao * nao * sizeof(double), stream);

#endif
    
#ifdef _SIMPLE_TIMER
    t_array_jk[5] += omp_get_wtime() - t2;
#endif
  
  for(int indxK=0; indxK<nset; ++indxK) {

    py::array_t<double> _dms = static_cast<py::array_t<double>>(_dms_list[indxK]); // element of 3D array (nset, nao, nao)
    py::buffer_info info_dms = _dms.request(); // 2D

    double * dms = static_cast<double*>(info_dms.ptr);

    pm->dev_push_async(d_dms, dms, nao*nao*sizeof(double), stream);
    
#ifdef _SIMPLE_TIMER
    t0 = omp_get_wtime();
#endif
    
    //    pm->dev_stream_wait(stream); // remove later

    {
      const double alpha = 1.0;
      const double beta = 0.0;
      const int nao2 = nao * nao;
      
      cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T, nao, nao, nao,
				&alpha, d_buf2, nao, nao2,
				d_dms, nao, 0,
				&beta, d_buf1, nao, nao2, naux);
    }
    
    // dgemm of (nao X blksize*nao) and (blksize*nao X nao) matrices - can refactor later...
    // vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))  // vk[k] is nao x nao array
  
    // buf3 = buf1.reshape(-1,nao).T
    // buf4 = buf2.reshape(-1,nao)

    //pm->dev_stream_wait(stream);

#ifdef _SIMPLE_TIMER
    double t1 = omp_get_wtime();
    t_array_jk[4] += t1 - t0;
#endif

    {
      dim3 grid_size((naux + (_TRANSPOSE_BLOCK_SIZE - 1)) / _TRANSPOSE_BLOCK_SIZE,
		     (nao + (_TRANSPOSE_BLOCK_SIZE - 1)) / _TRANSPOSE_BLOCK_SIZE, 1);
      dim3 block_size(_TRANSPOSE_BLOCK_SIZE, _TRANSPOSE_BLOCK_SIZE, 1);
      
      _getjk_transpose_buf1_buf3<<<grid_size, block_size, 0, stream>>>(d_buf3, d_buf1, naux, nao);
    }
    //    pm->dev_stream_wait(stream);
    
    // vk[k] += lib.dot(buf3, buf4)
    // gemm(A,B,C) : C = 1.0 * A.B + 0.0 * C
    // A is (m, k) matrix
    // B is (k, n) matrix
    // C is (m, n) matrix
    // Column-ordered: (A.B)^T = B^T.A^T
    
#ifdef _SIMPLE_TIMER
    double t3 = omp_get_wtime();
    t_array_jk[6] += t3 - t1;
#endif
    
    const double alpha = 1.0;
    const double beta = (count == 0) ? 0.0 : 1.0;
    
    const int m = nao; // # of rows of first matrix buf4^T
    const int n = nao; // # of cols of second matrix buf3^T
    const int k = naux*nao; // # of cols of first matrix buf4^

    const int lda = naux * nao;
    const int ldb = nao;
    const int ldc = (mode_getjk == 0) ? nset * nao: nao;
    
    const int vk_offset = (mode_getjk == 0) ? indxK * nao : indxK * nao*nao; // this is ugly...
    
#ifdef _CUDA_NVTX
    nvtxRangePushA("DGEMM");
#endif
    
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_buf2, ldb, d_buf3, lda, &beta, d_vkk+vk_offset, ldc);
    pm->dev_stream_wait(stream);
    
#ifdef _CUDA_NVTX
    nvtxRangePop();
#endif
   
#ifdef _SIMPLE_TIMER
    double t4 = omp_get_wtime();
    t_array_jk[7] += t4 - t3;
    t_array_jk_count++;
#endif 
  }
  
  //  printf("Leaving get_jk()\n");
}
  
/* ---------------------------------------------------------------------- */

void Device::fdrv(double *vout, double *vin, double *mo_coeff,
		  int nij, int nao, int *orbs_slice, int *ao_loc, int nbas, double * _buf)
{
//   const int ij_pair = nao * nao;
//   const int nao2 = nao * (nao + 1) / 2;
    
// #pragma omp parallel for
//   for (int i = 0; i < nij; i++) {
//     double * buf = &(_buf[i * nao * nao]);

//     int _i, _j, _ij;
//     double * tril = vin + nao2*i;
//     for (_ij = 0, _i = 0; _i < nao; _i++) 
//       for (_j = 0; _j <= _i; _j++, _ij++) {
// 	buf[_i*nao+_j] = tril[_ij];
// 	buf[_i+nao*_j] = tril[_ij]; // because going to use batched dgemm call on gpu
//       }
//   }
  
// #pragma omp parallel for
//   for (int i = 0; i < nij; i++) {
//     double * buf = &(_buf[i * nao * nao]);
    
//     const double D0 = 0;
//     const double D1 = 1;
//     const char SIDE_L = 'L';
//     const char UPLO_U = 'U';

//     double * _vout = vout + ij_pair*i;
    
//     dsymm_(&SIDE_L, &UPLO_U, &nao, &nao, &D1, buf, &nao, mo_coeff, &nao, &D0, _vout, &nao);    
//   }
  
}

#endif