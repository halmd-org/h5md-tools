#include <Python.h>
#include <numpy/ndarrayobject.h>

// forward declaration
bool is_double_matrix(PyArrayObject *a);

PyObject *_static_structure_factor(PyObject *self, PyObject *args)
{
    PyArrayObject *q, *r;
    
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &q, &PyArray_Type, &r)
            || q == NULL || !is_double_matrix(q)
            || r == NULL || !is_double_matrix(r))  {
        return NULL;
    }

    // Check that objects are of equal 2nd dimension
    if (PyArray_DIM(q, 2) != PyArray_DIM(r, 2)) {
        PyErr_SetString(PyExc_ValueError,
            "wavenumber array and coordinate array must be of equal 2nd dimension.");
        return NULL;
    }
    unsigned dimension = PyArray_DIM(q, 2);

    // loop over wavevectors 
    double result = 0;
    for (unsigned i=0; i < PyArray_DIM(q, 1); i++) {
        // loop over particles
        double sin_sum, cos_sum = 0;
        for (unsigned j=0; j < PyArray_DIM(r, 1); j++) {
            // q_r = inner(q, r)
            double q_r = 0;
            for (unsigned k=0; k < dimension; k++) {
                q_r += *(double*)PyArray_GETPTR2(q, i, k) * *(double*)PyArray_GETPTR2(r, j, k);
            }
            double s, c;
            sincos(q_r, &s, &c);
            sin_sum += s;
            cos_sum += c;
        }

        result +=  sin_sum * sin_sum + cos_sum * cos_sum;
    }
    result /= PyArray_DIM(q, 1) * PyArray_DIM(r, 1);

    return Py_BuildValue("d", result);
}

bool is_double_matrix(PyArrayObject *a)
{   if (PyArray_TYPE(a) != NPY_DOUBLE || a->nd != 1)
    {
        PyErr_SetString(PyExc_ValueError,
            "array must be of type Float and 2-dimensional.");
        return false;
    }
    return true;
}

