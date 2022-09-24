# numc
<strong>FILES WORKED ON: numc.c, matrix_optimized.c</strong>

Numc is a python module written in C which allows you to create matrices and perform different operations on them. You can perform element wise addition and subtraction, matrix multiplication, raising square matrices to powers, element wise negation, and element wise absolute value. Different performance techniques were used to speed up operations including parallelism (pragma omp statements), loop unrolling, intel intrinsic functions, and cache blocking.

Matrix operations were developed in matrix_optimized.c while the python-c interface was written in numc.c.

<strong>Install by running the command:</strong> <br>
python setup.py install --record files.txt

<strong>How to create a numc.Matrix object:</strong>

import numc as nc

<p><strong><em>nc.Matrix(rows: int, cols: int)</em></strong> will create a matrix with rows rows and cols cols. All entries in this matrix are defaulted to 0.</p>
<p><strong><em>nc.Matrix(rows: int, cols: int, val: float)</em></strong> will create a matrix with rows rows and cols cols. All entries in this matrix will be initialized to val.</p>
<p><strong><em>nc.Matrix(rows: int, cols: int, lst: List[int])</em></strong> will create a matrix with rows rows and cols cols. lst must have length rows * cols, and entries of the matrix will be initialized to values of lst in a row-major order.</p>
<p><strong><em>nc.Matrix(lst: List[List[int]])</em></strong> will create a matrix with the same shape as the 2D lst (i.e. each list in lst is a row for this matrix)</p>

<strong>How to index a numc.Matrix object:</strong><br>
You can index into a matrix and change either the value of one single entry or an entire row. More specifically, mat[i] should give you the ith row of matrix. If mat has more than 1 column, mat[i] should also be of type numc.Matrix with (mat’s number of columns, 1) as its shape. In other words, mat[i] returns a column vector. This is to support 2D indexing of numc matrices.

If mat only has one column, then mat[i] will return a double. mat[i][j] should give you the entry at the ith row and jth column. If you are setting one single entry by indexing, the data type must be float or int. If you are setting an entire row of a matrix that has more than one column, you must provide a 1D list that has the same length as the number of columns of that matrix. Every element of this list must be either of type float or int.

Please note that if mat[i] has more than 1 entry, it will share data with mat, and changing mat[i] will result in a change in mat.
You can obtain the shape of a matrix or vector by accessing their "shape" attribute. This will return a tuple of (rows, cols).<br>
Example: print(mat1.shape) # this should print out (rows, cols)

<strong>Available Methods:</strong>
<p><strong><em>a + b</em></strong>: Element-wise sum of a and b. a and b must have the same dimensions. Returns a numc.Matrix object.</p>
<p><strong><em>a - b</em></strong>: Element-wise subtraction of a and b. a and b must have the same dimensions. Returns a numc.Matrix object.</p>
<p><strong><em>a * b</em></strong>: Matrix multiplication of a and b. a’s number of columns must be equal to b’s number of rows. Returns a numc.Matrix object.</p>
<p><strong><em>-a</em></strong>: Element-wise negation of a. Returns a numc.Matrix object.</p>
<p><strong><em>abs(a)</em></strong>: Element-wise absolute value of a. Returns a numc.Matrix object.</p>
<p><strong><em>a ** pow</em></strong>: Raise a to the powth power. pow must be a positive integer and a must be a square matrix. Returns a numc.Matrix object.</p>