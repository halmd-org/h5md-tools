#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_mdplot_ext
#include <numpy/ndarrayobject.h>

PyObject * _static_structure_factor(PyObject *self, PyObject *args);

static PyMethodDef methods[] = {
    {"_static_structure_factor", _static_structure_factor, METH_VARARGS,
     "Compute static structure factor for given wavevectors and particle coordinates."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initext(void)
{
        (void) Py_InitModule("ext", methods);
        import_array();
}

