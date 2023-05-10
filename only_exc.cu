#include <iostream>
#include <cstdio>
#include <fstream>
#include <sys/mman.h>
#include <sstream>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

#include "matrix_op.h"
#include "my_matrix_op.h"
#include "libxc/include/xc.h"



#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
#define CUDA_TIME_DECLARE(val) cudaEvent_t start_##val, stop_##val; float time_##val; cudaEventCreate(&start_##val); cudaEventCreate(&stop_##val);
#define CUDA_TIME_RECORD(val, func)   cudaEventRecord(start_##val, 0); func; cudaEventRecord(stop_##val, 0); 
#define CUDA_TIME_SYNC(val) cudaEventSynchronize(stop_##val); cudaEventElapsedTime(&time_##val, start_##val, stop_##val); cudaEventDestroy(start_##val); cudaEventDestroy(stop_##val);

const int n_grid_blk = 131072;
const int n_bf_ = 190;
const int n_prim = 230;
const int n_shell = 100;
const int grids_size = 343100;
const int BLOCK_SIZE_ = 131072;
const int N_grid_ = 343100;

const int x_func_code = 450;
const int c_func_code = 236;

double *h_grid_ao_, *h_rho_, *h_sigma_, *h_lapl_, *h_tau_, *h_exc_, *h_vrho_, *h_D_, *h_vtau_;
double *grid_ao_, *rho_, *sigma_, *lapl_, *tau_, *exc_, *vrho_, *vsigma_, *D_, *vxc_, *vtau_, *vlapl_;
double *buf1_, *buf2_, *buf3_, *buf4_; 
double *h_alpha, *h_O, *h_coeff, *h_grid_r;
double *alpha, *O, *coeff, *grid_r;
double *h_weights_, *weights_;
__int8_t *h_ang, *ang;
int *h_sh_nprim, *sh_nprim;

xc_func_type* func_x_; // xc function
xc_func_type* func_c_; // xc function

double exc_total_ = 0;

// template<typename T>
// void check(T result, char const* const func, const char* const file, int const line) {
//   if (result) {
//     fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
//     exit(EXIT_FAILURE);
//   }
// }

__int64_t get_block_offset(int block_id) { return 1LL * block_id * BLOCK_SIZE_; }

int get_block_size(int block_id) {
  
  __int64_t st = get_block_offset(block_id);
  if (st + BLOCK_SIZE_ < grids_size) {
    return BLOCK_SIZE_;
  } else {
    return grids_size - st;
  }
}

double* get_block_weight(int block_id) {
  return weights_ + get_block_offset(block_id);
}

template<int MAX_NUM_PRIMITIVE>
__global__ void evaluate_grid_ao_deriv2_kernel(const int n_grids, const double* grid_r, double* grid_ao,
                                                double* O, double* coeff, double* alpha, __int8_t* ang, int* sh_nprim) {
  int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (grid_idx >= n_grids)
    return;

  double gx = grid_r[grid_idx * 3];
  double gy = grid_r[grid_idx * 3 + 1];
  double gz = grid_r[grid_idx * 3 + 2];

  double pow_rx[8] = {1}, pow_ry[8] = {1}, pow_rz[8] = {1};
  double va[MAX_NUM_PRIMITIVE], vc[MAX_NUM_PRIMITIVE];
  double out[10];

  for (int i = 0, co_idx = 0, bf_idx = 0; i < n_shell; i++) {
    double ox = O[i * 3];
    double oy = O[i * 3 + 1];
    double oz = O[i * 3 + 2];
    int s_ang = ang[i];
    int sh_coeff = sh_nprim[i];
    for (int j = 0; j < sh_coeff; j++) {
      va[j] = alpha[co_idx];
      vc[j] = coeff[co_idx];
      co_idx++;
    }

    double rx = gx - ox;
    double ry = gy - oy;
    double rz = gz - oz;

    double rx2 = rx * rx;
    double ry2 = ry * ry;
    double rz2 = rz * rz;

    for (int j = 1; j <= s_ang; j++) {
      pow_rx[j] = pow_rx[j - 1] * rx;
    }
    for (int j = 1; j <= s_ang; j++) {
      pow_ry[j] = pow_ry[j - 1] * ry;
    }
    for (int j = 1; j <= s_ang; j++) {
      pow_rz[j] = pow_rz[j - 1] * rz;
    }

    for (int l = s_ang; l >= 0; l--) {
      for (int m = s_ang - l; m >= 0; m--) {
        int n = s_ang - l - m;

        memset(out, 0, sizeof(double) * 10);
        for (int j = 0; j < sh_coeff; j++) {
          double v_alpha = va[j];
          double v_coeff = vc[j];

          double ear = exp(-v_alpha * (rx2 + ry2 + rz2));
          // first stage
          double dxy = -pow_rz[n] * ear;
          double dxz = -pow_ry[m] * ear;
          double dyz = -pow_rx[l] * ear;

          // second stage
          double dx = -pow_ry[m] * dxy, dxx = dx;
          double dy = -pow_rx[l] * dxy, dyy = dy;
          double dz = -pow_rx[l] * dxz, dzz = dz;

          out[0] += pow_rx[l] * dx * v_coeff;
          double tmp = 0;
          if (l < 1) {
            tmp = -2 * v_alpha * rx;
            dx *= tmp;
            dxz *= tmp;
            dxy *= tmp;
          } else {
            tmp = pow_rx[l - 1] * (l - 2 * v_alpha * rx2);
            dx *= tmp;
            dxz *= tmp;
            dxy *= tmp;
          }

          if (m < 1) {
            tmp = -2 * v_alpha * ry;
            dy *= tmp;
            dyz *= tmp;
            dxy *= -tmp;
          } else {
            tmp = pow_ry[m - 1] * (m - 2 * v_alpha * ry2);
            dy *= tmp;
            dyz *= tmp;
            dxy *= -tmp;
          }

          if (n < 1) {
            tmp = 2 * v_alpha * rz;
            dz *= -tmp;
            dyz *= tmp;
            dxz *= tmp;
          } else {
            tmp = pow_rz[n - 1] * (2 * v_alpha * rz2 - n);
            dz *= -tmp;
            dyz *= tmp;
            dxz *= tmp;
          }

          out[1] += dx * v_coeff;
          out[2] += dy * v_coeff;
          out[3] += dz * v_coeff;

          double a2 = v_alpha * v_alpha;
          if (l < 2) {
            dxx *= 4 * a2 * pow_rx[l] * rx2 - 2 * v_alpha * (2 * l + 1) * pow_rx[l];
          } else {
            dxx *= pow_rx[l - 2] * (4 * a2 * rx2 * rx2 - 2 * v_alpha * (2 * l + 1) * rx2 + (l * l - l));
          }
          if (m < 2) {
            dyy *= 4 * a2 * pow_ry[m] * ry2 - 2 * v_alpha * (2 * m + 1) * pow_ry[m];
          } else {
            dyy *= pow_ry[m - 2] * (4 * a2 * ry2 * ry2 - 2 * v_alpha * (2 * m + 1) * ry2 + (m * m - m));
          }
          if (n < 2) {
            dzz *= 4 * a2 * pow_rz[n] * rz2 - 2 * v_alpha * (2 * n + 1) * pow_rz[n];
          } else {
            dzz *= pow_rz[n - 2] * (4 * a2 * rz2 * rz2 - 2 * v_alpha * (2 * n + 1) * rz2 + (n * n - n));
          }

          out[4] += dxx * v_coeff;
          out[5] += dyy * v_coeff;
          out[6] += dzz * v_coeff;
          out[7] += dxy * v_coeff;
          out[8] += dxz * v_coeff;
          out[9] += dyz * v_coeff;
        }
        __syncthreads();
        double* p = grid_ao + bf_idx * n_grids + grid_idx;
        for (int j = 0; j < 10; j++) {
          *p = out[j];
          p += n_grids * n_bf_;
        }
        bf_idx++;
      }
    }
  }
}

template <typename T>
T parse_number(char*& p);

template <>
double parse_number<double>(char*& p) {
    return strtod(p, &p);
}

template <>
float parse_number<float>(char*& p) {
    return strtof(p, &p);
}

template <>
int parse_number<int>(char*& p) {
    return strtol(p, &p, 10);
}

template <>
__int8_t parse_number<__int8_t>(char*& p) {
    return static_cast<__int8_t>(strtol(p, &p, 10));
}

template <typename T>
void read_file(const char* filename, T* array, int size) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    struct stat sb;
    if (fstat(fd, &sb)== -1) {
    std::cerr << "Error getting file size: " << filename << std::endl;
    exit(1);
  }

  void* addr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
      std::cerr << "Error mmaping file: " << filename << std::endl;
      exit(1);
  }

  close(fd);

  char* p = static_cast<char*>(addr);
  for (int i = 0; i < size; ++i) {
      while (*p == ' ' || *p == '\n') {
          ++p;
      }
      array[i] = parse_number<T>(p);
  }

  if (munmap(addr, sb.st_size) == -1) {
      std::cerr << "Error munmaping file: " << filename << std::endl;
      exit(1);
  }
}

void evaluate_grid_ao(int block_id, double* grid_ao)
{
  __int64_t offset = get_block_offset(block_id);
  int n_grid_blk = get_block_size(block_id);

  const int BLOCK = 256;

  dim3 block((n_grid_blk - 1) / BLOCK + 1), threads(BLOCK);
  evaluate_grid_ao_deriv2_kernel<6><<<block, threads>>>(n_grid_blk, grid_r + offset * 3, buf4_,O,coeff,alpha,ang,sh_nprim);
  CUDA_CHECK(cudaGetLastError());
  matrix_op::transpose(buf4_, {10, n_bf_, n_grid_blk}, grid_ao);
}

void init_data()
{
    h_ang = (__int8_t*)malloc(sizeof(__int8_t) * n_shell);
    h_sh_nprim = (int*)malloc(sizeof(int) * n_shell);
    h_O = (double*)malloc(3 * n_shell * sizeof(double));
    h_alpha = (double*)malloc(n_prim * sizeof(double));
    h_coeff = (double*)malloc(n_prim * sizeof(double));
    h_D_ = (double*)malloc(sizeof(double) * n_bf_ * n_bf_);
    h_weights_ = (double*)malloc(sizeof(double) * grids_size);
    h_grid_r = (double*)malloc(3 * sizeof(double) * grids_size);

    CUDA_CHECK(cudaMalloc(&grid_ao_, 10 * n_grid_blk * n_bf_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&D_, n_bf_ * n_bf_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&rho_, 4 * n_grid_blk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&sigma_, n_grid_blk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&lapl_, n_grid_blk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tau_, n_grid_blk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&exc_, n_grid_blk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&vrho_, n_grid_blk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&vsigma_, n_grid_blk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&vlapl_, n_grid_blk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&weights_, grids_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&vtau_, n_grid_blk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&vxc_, n_bf_ * n_bf_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ang, n_shell * sizeof(__int8_t)));
    CUDA_CHECK(cudaMalloc(&sh_nprim, n_shell * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&O, 3 * n_shell * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&alpha, n_prim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&coeff, n_prim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&grid_r, 3 * grids_size * sizeof(double)));

    printf("finish cudamalloc\n");

    read_file<double>("data/D_.txt", h_D_, n_bf_ * n_bf_);
    read_file<double>("data/weights_.txt", h_weights_, grids_size);
    read_file<__int8_t>("data/ang.txt", h_ang, n_shell);
    read_file<int>("data/sh_nprim.txt", h_sh_nprim, n_shell);
    read_file<double>("data/O.txt", h_O, 3 * n_shell);
    read_file<double>("data/alpha.txt", h_alpha, n_prim);
    read_file<double>("data/coeff.txt", h_coeff, n_prim);
    read_file<double>("data/grid_r.txt", h_grid_r, 3 * grids_size);

    printf("finish read file\n");

    CUDA_CHECK(cudaMemcpy(D_, h_D_, n_bf_ * n_bf_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights_, h_weights_, grids_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ang, h_ang, n_shell * sizeof(__int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sh_nprim, h_sh_nprim, n_shell * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(O, h_O, 3 * n_shell * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(alpha, h_alpha, n_prim * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(coeff, h_coeff, n_prim * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grid_r, h_grid_r, 3 * grids_size * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&buf1_, (n_grid_blk * 3 + n_bf_ * 3) * n_bf_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf2_, n_grid_blk * n_bf_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf3_, n_grid_blk * 5 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf4_, n_grid_blk * 10 * n_bf_ * sizeof(double)));

    CUDA_CHECK(cudaMallocManaged(&func_x_, sizeof(xc_func_type)));
    xc_func_init(func_x_, x_func_code, XC_UNPOLARIZED);
    CUDA_CHECK(cudaMallocManaged(&func_c_, sizeof(xc_func_type)));
    xc_func_init(func_c_, c_func_code, XC_UNPOLARIZED);
}

void prepare_block_exc_vxc(int block_id)
{
    const int n_grid_blk = get_block_size(block_id);
    CUDA_TIME_DECLARE(0);
    CUDA_TIME_DECLARE(1);
    CUDA_TIME_RECORD(0,evaluate_grid_ao(block_id, grid_ao_));
    
    auto rho = [&](int i) { return rho_ + i * n_grid_blk; };
    auto grid_ao = [&](int i) { return grid_ao_ + i * n_grid_blk * n_bf_; };
    cudaEventRecord(start_1,0);
    // prepare libxc input: rho, sigma, lapl, tau
    {
      // einsum("bi,ij,bj->b")
      // (grid_ao, D_, grid_ao)->rho_[0]
      matrix_op::bmm(grid_ao(0), {n_grid_blk, n_bf_}, D_, {n_bf_, n_bf_}, buf1_);
      matrix_op::bmm(buf1_, {n_grid_blk, 1, n_bf_}, grid_ao(0), {n_grid_blk, n_bf_, 1}, rho(0));

      // deriv1
      for (int i = 1; i <= 3; i++) {
        matrix_op::bmm_ex(2.0, buf1_, {n_grid_blk, 1, n_bf_}, grid_ao(i), {n_grid_blk, n_bf_, 1}, 0.0, rho(i));
      }

      for (int i = 1; i <= 3; i++) {
        // einsum("bi,ij,bj->b")
        // (grid_ao(i), D_^T, grid_ao(i))->tau_
        matrix_op::bmm(grid_ao(i), {n_grid_blk, n_bf_}, D_, {n_bf_, n_bf_}, buf2_);
        matrix_op::bmm_ex(1.0, buf2_, {n_grid_blk, 1, n_bf_}, grid_ao(i), {n_grid_blk, n_bf_, 1}, i == 1 ? 0.0 : 1.0,
                          tau_);
      }

      // sigma_ = (rho(1)^2 ^ rho(2)^2 + rho(3)^2)
      for (int i = 1; i <= 3; i++) {
        matrix_op::bmm_ex(1.0, rho(i), {n_grid_blk, 1, 1}, rho(i), {n_grid_blk, 1, 1}, i == 1 ? 0.0 : 1.0, sigma_);
      }

      // sum of ao(dxx), ao(dyy), ao(dzz)
      matrix_op::add(grid_ao(4), grid_ao(5), {n_grid_blk, n_bf_}, buf2_);
      matrix_op::add(grid_ao(6), buf2_, {n_grid_blk, n_bf_}, buf2_);
      matrix_op::bmm(buf1_, {n_grid_blk, 1, n_bf_}, buf2_, {n_grid_blk, n_bf_, 1}, lapl_);
      // lapl_ = (lapl_ + tau_) * 2;
      matrix_op::add_ex(2.0, lapl_, {1, n_grid_blk}, 2.0, tau_, {1, n_grid_blk}, lapl_);
      // tau_ = tau_ * 0.5
      matrix_op::scale(tau_, {1, n_grid_blk}, 0.5);
    }
    cudaEventRecord(stop_1,0);
    CUDA_TIME_SYNC(0);
    CUDA_TIME_SYNC(1);
    printf("time0:%f, time1:%f\n", time_0, time_1);


    // run libxc
  {
    xc_mgga_exc_vxc(func_x_, n_grid_blk, rho_, sigma_, lapl_, tau_, exc_, vrho_, vsigma_, vlapl_, vtau_);
    if (func_c_ != nullptr) {
      auto buf = [&](int i) { return buf3_ + i * n_grid_blk; };
      xc_mgga_exc_vxc(func_c_, n_grid_blk, rho_, sigma_, lapl_, tau_, buf(0), buf(1), buf(2), buf(3), buf(4));

      matrix_op::add(exc_, buf(0), {1, n_grid_blk}, exc_);
      matrix_op::add(vrho_, buf(1), {1, n_grid_blk}, vrho_);
      matrix_op::add(vsigma_, buf(2), {1, n_grid_blk}, vsigma_);
      matrix_op::add(vlapl_, buf(3), {1, n_grid_blk}, vlapl_);
      matrix_op::add(vtau_, buf(4), {1, n_grid_blk}, vtau_);
    }
  }
}

void compute_block_xc(int block_id)
{
    prepare_block_exc_vxc(block_id);
    printf("finish prepare_block\n");
    
    const int n_grid_blk = get_block_size(block_id);
    double* grid_w = get_block_weight(block_id);

    auto rho = [&](int i) { return rho_ + i * n_grid_blk; };
    auto grid_ao = [&](int i) { return grid_ao_ + i * n_grid_blk * n_bf_; };

    // compute vxc
  //LDA and GGA part contribution
  matrix_op::bmm(grid_w, {n_grid_blk, 1, 1}, vrho_, {n_grid_blk, 1, 1}, buf1_);
  matrix_op::bmm(grid_ao(0), {n_grid_blk, n_bf_, 1}, buf1_, {n_grid_blk, 1, 1}, buf2_);

  // (grid_w * vsigma * 4)
  matrix_op::bmm_ex(4.0, grid_w, {n_grid_blk, 1, 1}, vsigma_, {n_grid_blk, 1, 1}, 0.0, buf1_);
  for (int i = 1; i <= 3; i++) {
    // (grid_w * vsigma * 4) * rho(i)
    matrix_op::bmm(buf1_, {n_grid_blk, 1, 1}, rho(i), {n_grid_blk, 1, 1}, buf3_);
    matrix_op::bmm_ex(1.0, grid_ao(i), {n_grid_blk, n_bf_, 1}, buf3_, {n_grid_blk, 1, 1}, 1.0, buf2_);
  }

  matrix_op::bmm_ex<double>(1.0, buf2_, {1, n_grid_blk, n_bf_, true}, grid_ao(0), {1, n_grid_blk, n_bf_}, 1.0, vxc_);

  // MGGA part contribution
  matrix_op::bmm_ex<double>(0.5, grid_w, {n_grid_blk, 1, 1}, vtau_, {n_grid_blk, 1, 1}, 0.0, buf1_);
  for (int i = 1; i <= 3; i++) {
    matrix_op::bmm(grid_ao(i), {n_grid_blk, n_bf_, 1}, buf1_, {n_grid_blk, 1, 1}, buf2_);
    matrix_op::bmm_ex(1.0, buf2_, {1, n_grid_blk, n_bf_, true}, grid_ao(i), {1, n_grid_blk, n_bf_}, 1.0, vxc_);
  }

  // symmetrization
  matrix_op::transpose(vxc_, {n_bf_, n_bf_}, buf1_);
  matrix_op::add_ex(0.5, vxc_, {n_bf_, n_bf_}, 0.5, buf1_, {n_bf_, n_bf_}, vxc_);

  // compute exc_total
  // einsum("b,b,b->")
  // (grid_w, rho_, exc_)
  matrix_op::bmm(grid_w, {n_grid_blk, 1, 1}, rho(0), {n_grid_blk, 1, 1}, buf1_);

  double exc_blk = 0.0;
  matrix_op::bmm(buf1_, {1, n_grid_blk}, exc_, {n_grid_blk, 1}, buf2_);
  CUDA_CHECK(cudaMemcpy(&exc_blk, buf2_, sizeof(exc_blk), cudaMemcpyDeviceToHost));
  printf("exc_blk: %.10lf\n",exc_blk);
  exc_total_ += exc_blk;
}

void warming_up()
{
  matrix_op::bmm(grid_ao_, {n_grid_blk, n_bf_}, D_, {n_bf_, n_bf_}, buf1_);
}

void fp_ratio(double* data, int num)
{
  printf("num:%d\n", num);
    int fp32_st = 0, fp16_st = 0, bf16_st = 0;
    double max = 0.0;
    for(int i = 0; i < num; i++)
    {
      double t = data[i];
      if(t < 1e-2) fp32_st++;
      if(t < 1e-5) fp16_st++;
      if(t< 1e-7) bf16_st++;
      if(t > 0.0)
      {
        if(t > max) max = t;
      }
      else
      {
        if(-t > max) max = -t;
      }
    }
    printf("fp32_st:%d, fp16_st:%d, bf16_st:%d\n", fp32_st, fp16_st, bf16_st);
    printf("fp32 ratio:%.2lf, fp16 ratio:%.2lf, bf16 ratio:%.2lf\n", fp32_st * 1.0 / num, fp16_st * 1.0 / num, bf16_st * 1.0 / num);
    printf("max:%.10lf\n", max);
}

void calculate_ratio()
{
  int block_id = 2;
    const int n_grid_blk = get_block_size(block_id);
    evaluate_grid_ao(block_id, grid_ao_);
    h_grid_ao_ = (double*)malloc(10 * n_grid_blk * n_bf_ * sizeof(double));
    cudaMemcpy(h_grid_ao_, grid_ao_, 10 * n_grid_blk * n_bf_ * sizeof(double), cudaMemcpyDeviceToHost);
    auto grid_ao = [&](int i) { return h_grid_ao_ + i * n_grid_blk * n_bf_; };
    int num = n_grid_blk * n_bf_;
    printf("grid_ao num:%d\n", num);
    int fp32_st = 0, fp16_st = 0, bf16_st = 0;
    double max = 0.0;
    for(int i = 0; i < num; i++)
    {
      double t = grid_ao(block_id)[i];
      if(t < 1e-2) fp32_st++;
      if(t < 1e-5) fp16_st++;
      if(t< 1e-7) bf16_st++;
      if(t > 0.0)
      {
        if(t > max) max = t;
      }
      else
      {
        if(-t > max) max = -t;
      }
    }
    printf("fp32_st:%d, fp16_st:%d, bf16_st:%d\n", fp32_st, fp16_st, bf16_st);
    printf("fp32 ratio:%.2lf, fp16 ratio:%.2lf, bf16 ratio:%.2lf\n", fp32_st * 1.0 / num, fp16_st * 1.0 / num, bf16_st * 1.0 / num);
    printf("max:%.10lf\n", max);
    printf("D_ ");
    fp_ratio(h_D_, n_bf_ * n_bf_);
    double* h_grid_w = h_weights_ + get_block_offset(block_id);
    printf("grid_w ");
    fp_ratio(h_grid_w, n_grid_blk);
}

void calc_origin_bmm_perf()
{
  int block_id = 0;
    int iter = 1;
    const int n_grid_blk = get_block_size(block_id);
    evaluate_grid_ao(block_id, grid_ao_);

    auto rho = [&](int i) { return rho_ + i * n_grid_blk; };
    auto grid_ao = [&](int i) { return grid_ao_ + i * n_grid_blk * n_bf_; };

    matrix_op::bmm(grid_ao(0), {n_grid_blk, n_bf_}, D_, {n_bf_, n_bf_}, buf1_);
    CUDA_TIME_DECLARE(0);
    cudaEventRecord(start_0, 0);
    for(int i = 0; i < iter; i++)
    {
      matrix_op::bmm(buf1_, {n_grid_blk, 1, n_bf_}, grid_ao(0), {n_grid_blk, n_bf_, 1}, rho(0));
    }
    cudaEventRecord(stop_0, 0);
    CUDA_TIME_SYNC(0);
    double gflops = 2 * (double)n_grid_blk * n_bf_ * iter / time_0 / 1e6;
    printf("origin batch vetctor_mul GFlops/s: %.2f time:%fms\n", gflops, (double)time_0/iter);
}

int main()
{
    cudaSetDevice(0);
    init_data();
    // printf("finish init_data\n");
    // warming_up();
    // printf("finish warming_up\n");
    // cudaDeviceSynchronize();
    // printf("start compute block\n");
    // prepare_block_exc_vxc(0);
    // for(int i = 0; i < 3; i++)
    // {
    //   compute_block_xc(i);
    // }
    calc_origin_bmm_perf();




}
