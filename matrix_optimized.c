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
            if (((*mat)->data = (double*)calloc(rows * cols, sizeof(double)))) {
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
        #pragma omp parallel for /* parallelization */
        for (int i = 0; i < data_size/4 * 4; i += 4) {
            mat_data[i] = val;   /* loop unrolling */
            mat_data[i + 1] = val;
            mat_data[i + 2] = val;
            mat_data[i + 3] = val;
        }
            
        for (int i = data_size/4 * 4; i < data_size; i++) {
            mat_data[i] = val;
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
            
            int data_size = rows * cols;
            __m256d vector1;
            __m256d vector2;
            __m256d results;
            #pragma omp parallel for private(vector1, vector2, results) /* parallelization, "private" prevents threads from overwriting each other */
            for (int i = 0; i < data_size/16 * 16; i += 16) {
                vector1 = _mm256_loadu_pd(mat1->data + i); /* loop unrolling, SIMD instructions */
                vector2 = _mm256_loadu_pd(mat2->data + i); /* 4 doubles per __m256d variable */
                results = _mm256_add_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i, results);

                vector1 = _mm256_loadu_pd(mat1->data + i + 4); 
                vector2 = _mm256_loadu_pd(mat2->data + i + 4);
                results = _mm256_add_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i + 4, results);

                vector1 = _mm256_loadu_pd(mat1->data + i + 8); 
                vector2 = _mm256_loadu_pd(mat2->data + i + 8);
                results = _mm256_add_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i + 8, results);

                vector1 = _mm256_loadu_pd(mat1->data + i + 12); 
                vector2 = _mm256_loadu_pd(mat2->data + i + 12);
                results = _mm256_add_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i + 12, results);
                //result->data[i] = mat1->data[i] + mat2->data[i];
            }

            for (int i = data_size/16 * 16; i < data_size/4 * 4; i += 4) {
                vector1 = _mm256_loadu_pd(mat1->data + i); 
                vector2 = _mm256_loadu_pd(mat2->data + i);
                results = _mm256_add_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i, results);
            }

            for (int i = data_size/4 * 4; i < data_size; i++) {
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

            int data_size = rows * cols;
            __m256d vector1;
            __m256d vector2;
            __m256d results;
            #pragma omp parallel for private(vector1, vector2, results)
            for (int i = 0; i < data_size/16 * 16; i += 16) {
                vector1 = _mm256_loadu_pd(mat1->data + i); 
                vector2 = _mm256_loadu_pd(mat2->data + i);
                results = _mm256_sub_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i, results);

                vector1 = _mm256_loadu_pd(mat1->data + i + 4); 
                vector2 = _mm256_loadu_pd(mat2->data + i + 4);
                results = _mm256_sub_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i + 4, results);

                vector1 = _mm256_loadu_pd(mat1->data + i + 8); 
                vector2 = _mm256_loadu_pd(mat2->data + i + 8);
                results = _mm256_sub_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i + 8, results);

                vector1 = _mm256_loadu_pd(mat1->data + i + 12); 
                vector2 = _mm256_loadu_pd(mat2->data + i + 12);
                results = _mm256_sub_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i + 12, results);
                //result->data[i] = mat1->data[i] - mat2->data[i];
            }

            for (int i = data_size/16 * 16; i < data_size/4 * 4; i += 4) {
                vector1 = _mm256_loadu_pd(mat1->data + i); 
                vector2 = _mm256_loadu_pd(mat2->data + i);
                results = _mm256_sub_pd(vector1, vector2);
                _mm256_storeu_pd(result->data + i, results);
            }

            for (int i = data_size/4 * 4; i < data_size; i++) {
                result->data[i] = mat1->data[i] - mat2->data[i];
            }
            return 0;
        }
    }
    return -1;
}

int mul_lite(matrix *result, matrix *mat1, matrix *mat2) { /* no cache blocking used */
    int mat1_rows = mat1->rows;
    int mat1_cols = mat1->cols;

    int n;
    int mat2_rows = n = mat2->rows;
    int mat2_cols = mat2->cols;

    int result_cols = result->cols;
    
    double *dst = (double*) malloc(sizeof(double) * mat2_rows * mat2_cols); //FREE REMINDER, check
    //dst == mat2 transposed

    #pragma omp parallel for
    for (int c = 0; c < mat2_rows; c++) {
        for (int d = 0; d < mat2_cols; d++) {
            dst[d*mat2_rows + c] = mat2->data[c*mat2_cols + d]; /* Transpose mat2 into column-major order in order to use SIMD instructions */
        }
    }
        
    __m256d vector1;
    __m256d vector2;
    __m256d vector3;
    __m256d vector4;
    __m256d vector5;
    __m256d vector6;
    __m256d vector7;
    __m256d vector8;
    __m256d results1;
    double temp;
    double result_arr[4];
    
    #pragma omp parallel for private(vector1, vector2, vector3, vector4, vector5, vector6, vector7, vector8, results1, temp, result_arr)
    for (int i = 0; i < mat1_rows; i++) {
        for (int j = 0; j < mat2_cols; j++) { 
            results1 = _mm256_set1_pd(0);
            temp = 0;
            for (int k = 0; k < n/16 * 16; k += 16) {
                vector1 = _mm256_loadu_pd(mat1->data + (i * mat1_cols + k)); 
                vector2 = _mm256_loadu_pd(mat1->data + (i * mat1_cols + k + 4));
                vector3 = _mm256_loadu_pd(mat1->data + (i * mat1_cols + k + 8));
                vector4 = _mm256_loadu_pd(mat1->data + (i * mat1_cols + k + 12));

                vector5 = _mm256_loadu_pd(dst + (j * mat2_rows + k));
                vector6 = _mm256_loadu_pd(dst + (j * mat2_rows + k + 4));
                vector7 = _mm256_loadu_pd(dst + (j * mat2_rows + k + 8));
                vector8 = _mm256_loadu_pd(dst + (j * mat2_rows + k + 12));

                results1 = _mm256_fmadd_pd(vector1, vector5, results1);
                results1 = _mm256_fmadd_pd(vector2, vector6, results1);
                results1 = _mm256_fmadd_pd(vector3, vector7, results1);
                results1 = _mm256_fmadd_pd(vector4, vector8, results1);
            }
            for (int k = n/16 * 16; k < n; k++) {
                temp += mat1->data[i * mat1_cols + k] * dst[j * mat2_rows + k];
            }
            _mm256_storeu_pd(result_arr, results1);
            result->data[i * result_cols + j] = temp + result_arr[0] + result_arr[1] + result_arr[2] + result_arr[3];   
        }
    }
    free(dst);
    return 0;
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

            int t_blocksize = 100;
            double *dst = (double*) malloc(sizeof(double) * mat2_rows * mat2_cols); //FREE REMINDER, check
            #pragma omp parallel for
            for (int a = 0; a < mat2_rows; a += t_blocksize) { /* cache blocking */
                for (int b = 0; b < mat2_cols; b += t_blocksize) {
                    for (int c = a; c < a + t_blocksize && c < mat2_rows; c++) {
                        for (int d = b; d < b + t_blocksize && d < mat2_cols; d++) {
                            dst[d*mat2_rows + c] = mat2->data[c*mat2_cols + d];
                        }
                    }
                }
            }
        
            __m256d vector1;
            __m256d vector2;
            __m256d vector3;
            __m256d vector4;
            __m256d vector5;
            __m256d vector6;
            __m256d vector7;
            __m256d vector8;
            __m256d results1;
            double temp;
            double result_arr[4];
            int blocksize = 100;
            #pragma omp parallel for private(vector1, vector2, vector3, vector4, vector5, vector6, vector7, vector8, results1, temp, result_arr)
            for (int i = 0; i < mat1_rows; i += blocksize) {
                for (int j = 0; j < mat2_cols; j += blocksize) { 
                    for (int blki = i; (blki < i + blocksize) && (blki < mat1_rows); blki++) {
                        for (int blkj = j; (blkj < j + blocksize) && (blkj < mat2_cols); blkj++) {
                            results1 = _mm256_set1_pd(0);
                            temp = 0;
                            //result->data[blki * result_cols + blkj] = 0;
                            for (int k = 0; k < n/16 * 16; k += 16) {
                                vector1 = _mm256_loadu_pd(mat1->data + (blki * mat1_cols + k));
                                vector2 = _mm256_loadu_pd(mat1->data + (blki * mat1_cols + k + 4));
                                vector3 = _mm256_loadu_pd(mat1->data + (blki * mat1_cols + k + 8));
                                vector4 = _mm256_loadu_pd(mat1->data + (blki * mat1_cols + k + 12));

                                vector5 = _mm256_loadu_pd(dst + (blkj * mat2_rows + k));
                                vector6 = _mm256_loadu_pd(dst + (blkj * mat2_rows + k + 4));
                                vector7 = _mm256_loadu_pd(dst + (blkj * mat2_rows + k + 8));
                                vector8 = _mm256_loadu_pd(dst + (blkj * mat2_rows + k + 12));

                                results1 = _mm256_fmadd_pd(vector1, vector5, results1);
                                results1 = _mm256_fmadd_pd(vector2, vector6, results1);
                                results1 = _mm256_fmadd_pd(vector3, vector7, results1);
                                results1 = _mm256_fmadd_pd(vector4, vector8, results1);
                                //result->data[blki * result_cols + blkj] += mat1->data[blki * mat1_cols + k] * dst[blkj * mat2_rows + k];
                            }
                            for (int k = n/16 * 16; k < n; k++) {
                                temp += mat1->data[blki * mat1_cols + k] * dst[blkj * mat2_rows + k];
                            }
                            _mm256_storeu_pd(result_arr, results1);
                            result->data[blki * result_cols + blkj] = temp + result_arr[0] + result_arr[1] + result_arr[2] + result_arr[3];
                        }
                    }
                }
            }
            free(dst);
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
    int rows;
    int cols;
    if (result && mat && ((rows = result->rows) == mat->rows) && ((cols = result->cols) == mat->cols)) {
        matrix* temp_mat;
        matrix* s_mat;
        if (!allocate_matrix(&temp_mat, rows, cols) &&
            !allocate_matrix(&s_mat, rows, cols)) { //Deallocate reminder

            #pragma omp parallel for
            for (int i = 0; i < (rows * cols)/4 * 4; i += 4) { //pragma and unrolling
                s_mat->data[i] = mat->data[i];
                s_mat->data[i + 1] = mat->data[i + 1];
                s_mat->data[i + 2] = mat->data[i + 2];
                s_mat->data[i + 3] = mat->data[i + 3];
            }
            for (int i = (rows * cols)/4 * 4; i < (rows * cols); i++) {
                s_mat->data[i] = mat->data[i];
            }

            #pragma omp parallel for
            for (int i = 0; i < rows; i++) { //pragma and unrolling, identity matrix
                for (int j = 0; j < cols/4 * 4; j += 4) {
                    if (i != j) {
                        result->data[i*cols + j] = 0;
                    } else {
                        result->data[i*cols + j] = 1;
                    }

                    if (i != (j + 1)) {
                        result->data[i*cols + j + 1] = 0;
                    } else {
                        result->data[i*cols + j + 1] = 1;
                    }

                    if (i != (j + 2)) {
                        result->data[i*cols + j + 2] = 0;
                    } else {
                        result->data[i*cols + j + 2] = 1;
                    }

                    if (i != (j + 3)) {
                        result->data[i*cols + j + 3] = 0;
                    } else {
                        result->data[i*cols + j + 3] = 1;
                    }
                }

                for (int j = cols/4 * 4; j < cols; j++) {
                    if (i != j) {
                        result->data[i*cols + j] = 0;
                    } else {
                        result->data[i*cols + j] = 1;
                    }
                }
            }
            
            int bit;
            double* temp_data; // repeated squaring algorithm
            for (int power = pow; power > 0; power = power/2) { //a = result, s = s_mat
                bit = power % 2;
                if (bit) {
                    mul_lite(temp_mat, result, s_mat);
                    temp_data = temp_mat->data;
                    temp_mat->data = result->data;
                    result->data = temp_data;
                }
                mul_lite(temp_mat, s_mat, s_mat);
                temp_data = temp_mat->data;
                temp_mat->data = s_mat->data;
                s_mat->data = temp_data;
            }

            deallocate_matrix(s_mat);
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

            int data_size = rows * cols;
            __m256d negator = _mm256_set1_pd(-1);
            __m256d vector1;
            __m256d results;
            #pragma omp parallel for private(vector1, results)
            for (int i = 0; i < data_size/16 * 16; i += 16) {
                vector1 = _mm256_loadu_pd(mat->data + i);
                results = _mm256_mul_pd(vector1, negator);
                _mm256_storeu_pd(result->data + i, results);

                vector1 = _mm256_loadu_pd(mat->data + i + 4);
                results = _mm256_mul_pd(vector1, negator);
                _mm256_storeu_pd(result->data + i + 4, results);

                vector1 = _mm256_loadu_pd(mat->data + i + 8);
                results = _mm256_mul_pd(vector1, negator);
                _mm256_storeu_pd(result->data + i + 8, results);

                vector1 = _mm256_loadu_pd(mat->data + i + 12);
                results = _mm256_mul_pd(vector1, negator);
                _mm256_storeu_pd(result->data + i + 12, results);
                //result->data[i] = mat->data[i] * -1;
            }

            for (int i = data_size/16 * 16; i < data_size/4 * 4; i += 4) {
                vector1 = _mm256_loadu_pd(mat->data + i);
                results = _mm256_mul_pd(vector1, negator);
                _mm256_storeu_pd(result->data + i, results);
            }

            for (int i = data_size/4 * 4; i < data_size; i++) {
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

            int data_size = rows * cols;
            __m256d negator = _mm256_set1_pd (-1);
            __m256d vector;
            __m256d v_negated;
            __m256d results;
            #pragma omp parallel for private(vector, v_negated, results)
            for (int i = 0; i < data_size/16 * 16; i += 16) {
                vector = _mm256_loadu_pd(mat->data + i);
                v_negated = _mm256_mul_pd(negator, vector);
                results = _mm256_max_pd(vector, v_negated);
                _mm256_storeu_pd(result->data + i, results);

                vector = _mm256_loadu_pd(mat->data + i + 4);
                v_negated = _mm256_mul_pd(negator, vector);
                results = _mm256_max_pd(vector, v_negated);
                _mm256_storeu_pd(result->data + i + 4, results);

                vector = _mm256_loadu_pd(mat->data + i + 8);
                v_negated = _mm256_mul_pd(negator, vector);
                results = _mm256_max_pd(vector, v_negated);
                _mm256_storeu_pd(result->data + i + 8, results);

                vector = _mm256_loadu_pd(mat->data + i + 12);
                v_negated = _mm256_mul_pd(negator, vector);
                results = _mm256_max_pd(vector, v_negated);
                _mm256_storeu_pd(result->data + i + 12, results);
            }

            for (int i = data_size/16 * 16; i < data_size/4 * 4; i += 4) {
                vector = _mm256_loadu_pd(mat->data + i);
                v_negated = _mm256_mul_pd(negator, vector);
                results = _mm256_max_pd(vector, v_negated);
                _mm256_storeu_pd(result->data + i, results);
            }

            for (int i = data_size/4 * 4; i < data_size; i++) {
                result->data[i] = (mat->data[i] >= 0) ? mat->data[i] : (mat->data[i] * -1);
            }

            return 0;
        }
    }
    return -1;
}

