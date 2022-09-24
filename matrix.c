#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, double low, double high) {
    srand(42);
    for (int i = 0; i < result->rows * result->cols; i++) {
        result->data[i] = rand_double(low, high);
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) { //can we assume mat is not null? check
    if (mat) {
        if ((*mat = (matrix*)malloc(sizeof(matrix))) && (rows > 0) && (cols > 0)) {
            if (((*mat)->data = (double*)malloc(sizeof(double) * rows * cols))) {
                double* mat_data = (*mat)->data;
                int data_size = rows * cols;
                #pragma omp parallel /*multi threading*/
                {
                    int num_threads = omp_get_num_threads();
                    int thread_num = omp_get_thread_num();
                    int chunk = data_size/num_threads;
                    
                    int start = thread_num * chunk;
                    int end;

                    if (thread_num != (num_threads - 1)) {
                        end = start + chunk;
                    } else {
                        end = data_size;
                    }

                    for (int i = start; i < end/6 * 6; i += 6) { /*loop unrolling*/
                        mat_data[i] = 0;
                        mat_data[i + 1] = 0;
                        mat_data[i + 2] = 0;
                        mat_data[i + 3] = 0;
                        mat_data[i + 4] = 0;
                        mat_data[i + 5] = 0;
                    }
                    
                    for (int i = end/6 * 6; i < end; i++) {
                        mat_data[i] = 0;
                    }
                }
                (*mat)->rows = rows;
                (*mat)->cols = cols;
                (*mat)->ref_cnt = 1;
                (*mat)->deallocated = 0;
                (*mat)->parent = NULL;
                return 0;
            }
        }
    }
    return -1;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) { //can we assume mat is not null and from is not null? check
    if (mat && from) {
        if ((*mat = (matrix*)malloc(sizeof(matrix))) && (rows > 0) && (cols > 0)) {
            (*mat)->rows = rows;
            (*mat)->cols = cols;
            (*mat)->data = (from->data) + offset;
            (*mat)->ref_cnt = 1;
            (*mat)->deallocated = 0;
            (*mat)->parent = from;

            from->ref_cnt += 1;
            return 0;
        }
    }
    return -1;
}

/*
 * This function frees the matrix struct pointed to by `mat`. However, you need to make sure that
 * you only free the data if `mat` is not a slice and has no existing slices, or if `mat` is the
 * last existing slice of its parent matrix and its parent matrix has no other references.
 * You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    if (mat) {
        if (mat->parent == NULL) {
            if (mat->ref_cnt == 1) {
                free(mat->data);
                free(mat);
            }
            else {
                mat->deallocated = 1;
            }
        }
        else {
            if ((mat->parent->ref_cnt > 2) || !(mat->parent->deallocated)) {
                //remove and decrement parent ref_cnt
                mat->parent->ref_cnt -= 1;
                free(mat);
            }
            else {
                free(mat->parent->data);
                free(mat->parent);
                free(mat);
            }
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) { //is mat always non-null? no need to worry here
    int cols = mat->cols;
    return mat->data[row * cols + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) { //is mat always non null? check
    if (mat) {
        int cols = mat->cols;
        mat->data[row * cols + col] = val;
    }
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) { //mat? check
    if (mat) {
        int data_size = (mat->rows) * (mat->cols);
        double* mat_data = mat->data;
        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int thread_num = omp_get_thread_num();
            int chunk = data_size/num_threads;
            
            int start = thread_num * chunk;
            int end;

            if (thread_num != (num_threads - 1)) {
                end = start + chunk;
            } else {
                end = data_size;
            }

            for (int i = start; i < end/6 * 6; i += 6) {
                mat_data[i] = val;
                mat_data[i + 1] = val;
                mat_data[i + 2] = val;
                mat_data[i + 3] = val;
                mat_data[i + 4] = val;
                mat_data[i + 5] = val;
            }
            
            for (int i = end/6 * 6; i < end; i++) {
                mat_data[i] = val;
            }
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) { //check
    if (result && mat1 && mat2) {
        int rows = result->rows;
        int cols = result->cols;

        if ((cols == mat1->cols) && (cols == mat2->cols) &&
            (rows == mat1->rows) && (rows == mat2->rows)) {
            for (int i = 0; i < (rows * cols); i++) {
                result->data[i] = mat1->data[i] + mat2->data[i];
            }
            return 0;
        }
    }
    return -1;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) { //check
    if (result && mat1 && mat2) {
        int rows = result->rows;
        int cols = result->cols;

        if ((cols == mat1->cols) && (cols == mat2->cols) &&
            (rows == mat1->rows) && (rows == mat2->rows)) {
            for (int i = 0; i < (rows * cols); i++) {
                result->data[i] = mat1->data[i] - mat2->data[i];
            }
            return 0;
        }
    }
    return -1;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) { //check
    if (result && mat1 && mat2) {
        int mat1_rows = mat1->rows;
        int mat1_cols = mat1->cols;

        int n;
        int mat2_rows = n = mat2->rows;
        int mat2_cols = mat2->cols;

        int result_rows = result->rows;
        int result_cols = result->cols;

        if ((mat1_cols == mat2_rows) && (result_rows == mat1_rows) && (result_cols == mat2_cols)) {
            for (int i = 0; i < mat1_rows; i++) {
                for (int j = 0; j < mat2_cols; j++) {
                    result->data[i * result_cols + j] = 0;
                    for (int k = 0; k < n; k++) {
                        result->data[i * result_cols + j] += (mat1->data[i * mat1_cols + k]) * (mat2->data[k * mat2_cols + j]);
                    }
                }
            }
            return 0;
        }
    }
    return -1;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) { // check
    if (result && mat) {
        matrix* temp_mat;
        if (!allocate_matrix(&temp_mat, result->rows, result->cols)) { //Deallocate reminder (check)

            for (int i = 0; i < ((result->rows) * (result->cols)); i++) {
                result->data[i] = mat->data[i];
            }

            for (int i = 2; i <= pow; i++) {
                if (mul_matrix(temp_mat, result, mat)) { // check to make sure operation is valid. 
                    return -1;
                }
                double* temp_data = temp_mat->data;
                temp_mat->data = result->data;
                result->data = temp_data;
            }
                
            deallocate_matrix(temp_mat); //already takes care of freeing matrix data
            return 0;
        }
        
    }
    return -1;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) { //check
    if (result && mat) {
        int rows = result->rows;
        int cols = result->cols;
        if ((rows == mat->rows) && (cols == mat->cols)) {
            for (int i = 0; i < (rows * cols); i++) {
                result->data[i] = mat->data[i] * -1;
            }
            return 0;
        }
    }
    return -1;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) { //check
    if (result && mat) {
        int rows = result->rows;
        int cols = result->cols;
        if ((rows == mat->rows) && (cols == mat->cols)) {
            double curr_element;
            for (int i = 0; i < (rows * cols); i++) {
                curr_element = mat->data[i];
                result->data[i] = (curr_element >= 0) ? curr_element : (curr_element * -1);
            }
            return 0;
        }
    }
    return -1;
}

