import unittest
import numc as nc
import time

def spec_test():
    # This creates a 3 * 3 matrix with entries all zeros
    start_time = time.time()
    mat1 = nc.Matrix(3, 3)

    # This creates a 2 * 3 matrix with entries all ones
    mat2 = nc.Matrix(3, 3, 1)

    # This creates a 2 * 3 matrix with first row 1, 2, 3, second row 4, 5, 6
    mat3 = nc.Matrix([[1, 2, 3], [4, 5, 6]])

    # This creates a 1 * 2 matrix with entries 4, 5
    mat4 = nc.Matrix(1, 2, [4, 5])

    print("\nSpec test..........................................................")
    print(mat1[0]) # this gives the 0th row of mat1, should print out [[0.0], [0.0], [0.0]]
    print(mat1[0][1]) # this should print out 0
    mat1[0][1] = 5
    print(mat1) # this should print out [[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    mat1[0] = [4, 5, 6]
    print(mat1) # this should print out [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # mat2
    print("\n")
    print(mat2) # [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    mat2[1][1] = 2
    print(mat2[1]) # [[1.0], [2.0], [1.0]]

    # You can change a value in a slice, and that will change the original matrix
    print("\n")
    print(mat2) # [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
    mat2_slice = mat2[0] # [[1.0], [1.0], [1.0]]
    mat2_slice[0] = 5
    print(mat2_slice) # [[5.0], [1.0], [1.0]]
    print(mat2) # [[5.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]

    print("\n")
    mat2 = 4
    print(mat2_slice)
    print("\nTime =", time.time() - start_time)
    print("Spec test..........................................................\n")

def add_test():
    print("Add test..........................................................")
    mat1 = nc.Matrix(3, 4, 1)
    mat2 = nc.Matrix(3, 4, 1)
    result = mat1 + mat2
    print(result)
    print("Add test..........................................................\n")

def sub_test():
    print("Sub test..........................................................")
    mat1 = nc.Matrix(5, 7, 5)
    mat2 = nc.Matrix(5, 7, 2)
    result = mat1 - mat2
    print(result)
    print("Sub test..........................................................\n")

def multiply_test():
    print("Multiply test..........................................................")
    mat1 = nc.Matrix([[1, 2, 3], [4, 5, 6]])
    mat2 = nc.Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    result = mat1 * mat2
    print(result)
    print("Multiply test..........................................................\n")

def negation_test():
    print("Negation test..........................................................")
    mat1 = nc.Matrix([[1, 2, 3], [4, 5, 6]])
    result = -mat1
    print(result)
    print("Negation test..........................................................\n")

def abs_test():
    print("Absolute Value test..........................................................")
    mat1 = nc.Matrix([[-1, 2, -3], [-4, -5, 6]])
    print(abs(mat1))
    print("Absolute Value test..........................................................\n")

def pow_test():
    print("Power test..........................................................")
    mat1 = nc.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = mat1**4
    print(result)
    print(result.shape)
    print("Power test..........................................................\n")

def set_test():
    print("Set test..........................................................")
    mat1 = nc.Matrix([[1, 2, 3], [4, 5, 6]])
    mat1.set(0, 0, 0)
    mat1.set(0, 1, 0)
    mat1.set(1, 2, 0)
    print(mat1)
    print("Set test..........................................................\n")

def get_test():
    print("Get test..........................................................")
    mat1 = nc.Matrix([[1, 2, 3], [4, 5, 6]])
    a = mat1.get(0, 0)
    b = mat1.get(0, 1)
    c = mat1.get(1, 2)
    print(a, b, c)
    print("Get test..........................................................\n")

def _test():
    print(" test..........................................................")

    print(" test..........................................................\n")

if __name__ == "__main__":
    spec_test()
    add_test()
    sub_test()
    multiply_test()
    negation_test()
    abs_test()
    pow_test()
    set_test()
    get_test()