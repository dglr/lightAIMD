#ifndef MATRIX_OP_H_
#define MATRIX_OP_H_

#include <memory>

#include <cublasLt.h>

#define CUBLAS_CHECK(x)                                                                                                \
  {                                                                                                                    \
    const auto err = x;                                                                                                \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                                                \
      printf("Error status %d in file %s line %d\n", err, __FILE__, __LINE__);                                         \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  }

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void check(T result, char const* const func, const char* const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}



namespace matrix_op {

class CuBLASLTHelper {
public:
  static cublasLtHandle_t get_handle() { return get_instance()->handle_; }
  static size_t get_work_size() { return work_size_; }
  static void* get_workspace() { return get_instance()->workspace_; }

  ~CuBLASLTHelper() {
    CUDA_CHECK(cudaFree(workspace_));
    CUBLAS_CHECK(cublasLtDestroy(handle_));
  }

private:
  static std::shared_ptr<CuBLASLTHelper> get_instance() {
    static std::shared_ptr<CuBLASLTHelper> d(new CuBLASLTHelper);
    return d;
  }

  CuBLASLTHelper() {
    CUBLAS_CHECK(cublasLtCreate(&handle_));
    CUDA_CHECK(cudaMalloc(&workspace_, work_size_));
  }

  cublasLtHandle_t handle_;
  static const size_t work_size_ = (1ULL << 29); // 512M
  void* workspace_;
};

template<typename U>
struct CuBLASTypeTraits;

template<>
struct CuBLASTypeTraits<double> {
  static const cudaDataType_t cuda_type = CUDA_R_64F;
  static const cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;
  typedef double ScalarType;
};

template<>
struct CuBLASTypeTraits<float> {
  static const cudaDataType_t cuda_type = CUDA_R_32F;
  static const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  typedef float ScalarType;
};

template<typename ComputeType>
class MatrixDesc {
public:
  MatrixDesc(int batch, int n, int m, bool transpose) : batch_(batch), n_(n), m_(m), transpose_(transpose) {
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&desc_, CuBLASTypeTraits<ComputeType>::cuda_type, n_, m_, m_));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_, sizeof(batch_)));
    int64_t stride = 1LL * n_ * m_;
    CUBLAS_CHECK(
        cublasLtMatrixLayoutSetAttribute(desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  }
  MatrixDesc(int batch, int n, int m) : MatrixDesc(batch, n, m, false) {}
  MatrixDesc(int n, int m) : MatrixDesc(1, n, m, false) {}
  MatrixDesc(const MatrixDesc& o) : MatrixDesc(o.batch_, o.n_, o.m_, o.transpose_) {}

  ~MatrixDesc() { CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(desc_)); }

  cublasLtMatrixLayout_t get_cublasLt_desc() const { return desc_; }
  int get_batch() const { return batch_; }
  int get_n() const { return n_; }
  int get_m() const { return m_; }
  bool is_transpose() const { return transpose_; }

private:
  int batch_, n_, m_;
  bool transpose_;
  cublasLtMatrixLayout_t desc_;
};

// compute alpha*(AxB) + beta*C
// Example:
//    bmm_ex(2.0, a, {2, 3, 4}, b, {2, 4, 3}, 3.0, c) -> 2*(AxB) + 3.0*C
template<typename ComputeType>
void bmm_ex(const ComputeType alpha, const ComputeType* a, const MatrixDesc<ComputeType>& a_desc, const ComputeType* b,
            const MatrixDesc<ComputeType>& b_desc, const ComputeType beta, ComputeType* c) {
  MatrixDesc<ComputeType> c_desc(a_desc.get_batch(), a_desc.is_transpose() ? a_desc.get_m() : a_desc.get_n(),
                                 b_desc.is_transpose() ? b_desc.get_n() : b_desc.get_m());

  cublasLtMatmulDesc_t op_desc = nullptr;

  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, CuBLASTypeTraits<ComputeType>::compute_type,
                                        CuBLASTypeTraits<ComputeType>::cuda_type));

  auto op_a = (a_desc.is_transpose() ? CUBLAS_OP_T : CUBLAS_OP_N);
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
  auto op_b = (b_desc.is_transpose() ? CUBLAS_OP_T : CUBLAS_OP_N);
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));

  const size_t work_size = CuBLASLTHelper::get_work_size();
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &work_size,
                                                    sizeof(work_size)));

  int n_algo = 0;
  cublasLtMatmulHeuristicResult_t algo[1] = {};
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(CuBLASLTHelper::get_handle(), op_desc, a_desc.get_cublasLt_desc(),
                                              b_desc.get_cublasLt_desc(), c_desc.get_cublasLt_desc(),
                                              c_desc.get_cublasLt_desc(), preference, 1, algo, &n_algo));

  CUBLAS_CHECK(cublasLtMatmul(CuBLASLTHelper::get_handle(), op_desc, &alpha, a, a_desc.get_cublasLt_desc(), b,
                              b_desc.get_cublasLt_desc(), &beta, c, c_desc.get_cublasLt_desc(), c,
                              c_desc.get_cublasLt_desc(), &(algo[0].algo), CuBLASLTHelper::get_workspace(),
                              CuBLASLTHelper::get_work_size(), 0));

  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
}

// Usage example:
//    matmul(a, {2, 3, 4}, b, {2, 4, 3}, c), c will have shape(2, 3, 3)
template<typename ComputeType>
void bmm(const ComputeType* a, const MatrixDesc<ComputeType>& a_desc, const ComputeType* b,
         const MatrixDesc<ComputeType>& b_desc, ComputeType* c) {
  bmm_ex(1.0, a, a_desc, b, b_desc, 0.0, c);
}

// compute alpha*A + beta*B
template<typename ComputeType>
void add_ex(const ComputeType alpha, const ComputeType* a, const MatrixDesc<ComputeType>& a_desc,
            const ComputeType beta, const ComputeType* b, const MatrixDesc<ComputeType>& b_desc, ComputeType* c) {
  MatrixDesc<ComputeType> c_desc(a_desc.get_batch(), a_desc.is_transpose() ? a_desc.get_m() : a_desc.get_n(),
                                 a_desc.is_transpose() ? a_desc.get_n() : a_desc.get_m());

  cublasLtMatrixTransformDesc_t op_desc = nullptr;

  CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&op_desc, CuBLASTypeTraits<ComputeType>::cuda_type));

  auto op_a = (a_desc.is_transpose() ? CUBLAS_OP_T : CUBLAS_OP_N);
  CUBLAS_CHECK(
      cublasLtMatrixTransformDescSetAttribute(op_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op_a, sizeof(op_a)));
  auto op_b = (b_desc.is_transpose() ? CUBLAS_OP_T : CUBLAS_OP_N);
  CUBLAS_CHECK(
      cublasLtMatrixTransformDescSetAttribute(op_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &op_b, sizeof(op_b)));

  CUBLAS_CHECK(cublasLtMatrixTransform(CuBLASLTHelper::get_handle(), op_desc, &alpha, a, a_desc.get_cublasLt_desc(),
                                       &beta, b, b_desc.get_cublasLt_desc(), c, c_desc.get_cublasLt_desc(), 0));

  CUBLAS_CHECK(cublasLtMatrixTransformDescDestroy(op_desc));
}

// compute A + B
template<typename ComputeType>
void add(const ComputeType* a, const ComputeType* b, const MatrixDesc<ComputeType>& desc, ComputeType* c) {
  add_ex(1.0, a, desc, 1.0, b, desc, c);
}

// compute alpha * A  (inplace)
template<typename ComputeType>
void scale(ComputeType* a, const MatrixDesc<ComputeType>& desc, ComputeType scale) {
  ComputeType* b = nullptr;
  add_ex(scale, a, desc, 0.0, b, desc, a);
}

// transpose A
template<typename ComputeType>
void transpose(const ComputeType* a, const MatrixDesc<ComputeType>& desc, ComputeType* b) {
  double* p = nullptr;
  MatrixDesc<ComputeType> a_desc(desc.get_batch(), desc.get_n(), desc.get_m(), true);
  add_ex(1.0, a, a_desc, 0.0, p, desc, b);
}

} // namespace lcc::matrix_op

#endif // MATRIX_OP_H_
