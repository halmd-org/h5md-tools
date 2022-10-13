/* C extensions used by mdplot scripts
 *
 * Copyright © 2008-2010  Felix Höfling, Peter Colberg
 *
 * This file is part of mdplot.
 *
 * mdplot is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL Py_Array_API_mdplot_ext
#include <numpy/ndarrayobject.h>

PyObject * _static_structure_factor(PyObject *self, PyObject *args);

static PyMethodDef methods[] = {
    {"_static_structure_factor", _static_structure_factor, METH_VARARGS,
     "Compute static structure factor for given wavevectors and particle coordinates."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef ext =
{
    PyModuleDef_HEAD_INIT,
    "ext",       /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC PyInit_ext(void)
{
    import_array();
    return PyModule_Create(&ext);
}

