// python3 setup.py build_ext --inplace
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <gmp.h>

static int ensure_int64_2d(PyArrayObject* arr, npy_intp second_dim) {
    if (!arr) return 0;
    if (PyArray_NDIM(arr) != 2) return 0;
    npy_intp *dims = PyArray_DIMS(arr);
    if (dims[1] != second_dim) return 0;
    if (PyArray_TYPE(arr) != NPY_INT64) return 0;
    return 1;
}

static int ensure_int64_1d(PyArrayObject* arr) {
    if (!arr) return 0;
    if (PyArray_NDIM(arr) != 1) return 0;
    if (PyArray_TYPE(arr) != NPY_INT64) return 0;
    return 1;
}

static PyObject* dot_mod(PyObject* self, PyObject* args) {
    PyObject *xobj, *yobj;
    long long p;
    if (!PyArg_ParseTuple(args, "OOL", &xobj, &yobj, &p)) return NULL;
    PyArrayObject *xarr = (PyArrayObject*)PyArray_FROM_OTF(xobj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yarr = (PyArrayObject*)PyArray_FROM_OTF(yobj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!xarr || !yarr) { Py_XDECREF(xarr); Py_XDECREF(yarr); return NULL; }
    if (!ensure_int64_2d(xarr, 16) || !ensure_int64_2d(yarr, 16)) {
        PyErr_SetString(PyExc_ValueError, "X and Y must be 2D int64 with second dim 16");
        Py_DECREF(xarr); Py_DECREF(yarr); return NULL;
    }
    npy_intp rb = PyArray_DIMS(xarr)[0];
    npy_intp cb = PyArray_DIMS(yarr)[0];
    npy_intp out_dims[2] = { rb, cb };
    PyArrayObject *out = (PyArrayObject*)PyArray_SimpleNew(2, out_dims, NPY_INT64);
    if (!out) { Py_DECREF(xarr); Py_DECREF(yarr); return NULL; }
    long long *X = (long long*)PyArray_DATA(xarr);
    long long *Y = (long long*)PyArray_DATA(yarr);
    long long *O = (long long*)PyArray_DATA(out);
    npy_intp sx0 = PyArray_STRIDES(xarr)[0] / sizeof(long long);
    npy_intp sx1 = PyArray_STRIDES(xarr)[1] / sizeof(long long);
    npy_intp sy0 = PyArray_STRIDES(yarr)[0] / sizeof(long long);
    npy_intp sy1 = PyArray_STRIDES(yarr)[1] / sizeof(long long);
    npy_intp so0 = PyArray_STRIDES(out)[0] / sizeof(long long);
    npy_intp so1 = PyArray_STRIDES(out)[1] / sizeof(long long);
    #pragma omp parallel for collapse(2) if (rb * cb > 1024)
    for (npy_intp r = 0; r < rb; ++r) {
        for (npy_intp c = 0; c < cb; ++c) {
            long long s = 0;
            for (int k = 0; k < 16; ++k) {
                long long a = X[r * sx0 + k * sx1] % p; if (a < 0) a += p;
                long long b = Y[c * sy0 + k * sy1] % p; if (b < 0) b += p;
                unsigned long long prod = (unsigned long long)a * (unsigned long long)b;
                s = (long long)(( (unsigned long long)(s % p) + (prod % (unsigned long long)p) ) % (unsigned long long)p);
            }
            O[r * so0 + c * so1] = s;
        }
    }
    Py_DECREF(xarr); Py_DECREF(yarr);
    return (PyObject*)out;
}

static PyObject* dot_eq_mod_mask(PyObject* self, PyObject* args) {
    PyObject *xobj, *yobj;
    long long p;
    long long rhs;
    if (!PyArg_ParseTuple(args, "OOLL", &xobj, &yobj, &p, &rhs)) return NULL;
    PyArrayObject *xarr = (PyArrayObject*)PyArray_FROM_OTF(xobj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yarr = (PyArrayObject*)PyArray_FROM_OTF(yobj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (!xarr || !yarr) { Py_XDECREF(xarr); Py_XDECREF(yarr); return NULL; }
    if (!ensure_int64_2d(xarr, 16) || !ensure_int64_2d(yarr, 16)) {
        PyErr_SetString(PyExc_ValueError, "X and Y must be 2D int64 with second dim 16");
        Py_DECREF(xarr); Py_DECREF(yarr); return NULL;
    }
    npy_intp rb = PyArray_DIMS(xarr)[0];
    npy_intp cb = PyArray_DIMS(yarr)[0];
    npy_intp out_dims[2] = { rb, cb };
    PyArrayObject *out = (PyArrayObject*)PyArray_SimpleNew(2, out_dims, NPY_BOOL);
    if (!out) { Py_DECREF(xarr); Py_DECREF(yarr); return NULL; }
    long long *X = (long long*)PyArray_DATA(xarr);
    long long *Y = (long long*)PyArray_DATA(yarr);
    npy_bool *O = (npy_bool*)PyArray_DATA(out);
    npy_intp sx0 = PyArray_STRIDES(xarr)[0] / sizeof(long long);
    npy_intp sx1 = PyArray_STRIDES(xarr)[1] / sizeof(long long);
    npy_intp sy0 = PyArray_STRIDES(yarr)[0] / sizeof(long long);
    npy_intp sy1 = PyArray_STRIDES(yarr)[1] / sizeof(long long);
    npy_intp so0 = PyArray_STRIDES(out)[0] / sizeof(npy_bool);
    npy_intp so1 = PyArray_STRIDES(out)[1] / sizeof(npy_bool);
    #pragma omp parallel for collapse(2) if (rb * cb > 1024)
    for (npy_intp r = 0; r < rb; ++r) {
        for (npy_intp c = 0; c < cb; ++c) {
            long long s = 0;
            for (int k = 0; k < 16; ++k) {
                long long a = X[r * sx0 + k * sx1] % p; if (a < 0) a += p;
                long long b = Y[c * sy0 + k * sy1] % p; if (b < 0) b += p;
                unsigned long long prod = (unsigned long long)a * (unsigned long long)b;
                s = (long long)(( (unsigned long long)(s % p) + (prod % (unsigned long long)p) ) % (unsigned long long)p);
            }
            O[r * so0 + c * so1] = (s % p) == (rhs % p);
        }
    }
    Py_DECREF(xarr); Py_DECREF(yarr);
    return (PyObject*)out;
}

static PyObject* verify_first_match(PyObject* self, PyObject* args) {
    PyObject *xobj, *yobj, *xseeds_obj, *yseeds_obj, *alist_obj, *kobj, *rhsobj, *mask_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &xobj, &yobj, &xseeds_obj, &yseeds_obj, &alist_obj, &kobj, &rhsobj, &mask_obj)) return NULL;
    PyArrayObject *xarr = (PyArrayObject*)PyArray_FROM_OTF(xobj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yarr = (PyArrayObject*)PyArray_FROM_OTF(yobj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *xseeds = (PyArrayObject*)PyArray_FROM_OTF(xseeds_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yseeds = (PyArrayObject*)PyArray_FROM_OTF(yseeds_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *mask = (PyArrayObject*)PyArray_FROM_OTF(mask_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (!xarr || !yarr || !xseeds || !yseeds || !mask) { Py_XDECREF(xarr); Py_XDECREF(yarr); Py_XDECREF(xseeds); Py_XDECREF(yseeds); Py_XDECREF(mask); return NULL; }
    if (!ensure_int64_2d(xarr, 16) || !ensure_int64_2d(yarr, 16) || !ensure_int64_1d(xseeds) || !ensure_int64_1d(yseeds)) {
        PyErr_SetString(PyExc_ValueError, "invalid array shapes/dtypes");
        Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL;
    }
    if (PyArray_NDIM(mask) != 2) { PyErr_SetString(PyExc_ValueError, "mask must be 2D bool array"); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
    npy_intp rb = PyArray_DIMS(xarr)[0];
    npy_intp cb = PyArray_DIMS(yarr)[0];
    if (PyArray_DIMS(mask)[0] != rb || PyArray_DIMS(mask)[1] != cb) { PyErr_SetString(PyExc_ValueError, "mask shape mismatch"); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
    if (PyArray_DIMS(xseeds)[0] != rb || PyArray_DIMS(yseeds)[0] != cb) { PyErr_SetString(PyExc_ValueError, "seed arrays length mismatch"); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
    if (!PySequence_Check(alist_obj) || PySequence_Size(alist_obj) != 16) { PyErr_SetString(PyExc_ValueError, "a_mod_k must be length 16"); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
    PyObject* a_arr[16];
    for (int i = 0; i < 16; ++i) { PyObject* it = PySequence_GetItem(alist_obj, i); if (!it) { for (int j=0;j<i;++j) Py_DECREF(a_arr[j]); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; } a_arr[i] = it; }
    long long *X = (long long*)PyArray_DATA(xarr);
    long long *Y = (long long*)PyArray_DATA(yarr);
    long long *XS = (long long*)PyArray_DATA(xseeds);
    long long *YS = (long long*)PyArray_DATA(yseeds);
    npy_bool *M = (npy_bool*)PyArray_DATA(mask);
    npy_intp sx0 = PyArray_STRIDES(xarr)[0] / sizeof(long long);
    npy_intp sx1 = PyArray_STRIDES(xarr)[1] / sizeof(long long);
    npy_intp sy0 = PyArray_STRIDES(yarr)[0] / sizeof(long long);
    npy_intp sy1 = PyArray_STRIDES(yarr)[1] / sizeof(long long);
    npy_intp sm0 = PyArray_STRIDES(mask)[0] / sizeof(npy_bool);
    npy_intp sm1 = PyArray_STRIDES(mask)[1] / sizeof(npy_bool);
    PyObject* rhs_mod = PyNumber_Remainder(rhsobj, kobj);
    if (!rhs_mod) { for (int j=0;j<16;++j) Py_DECREF(a_arr[j]); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
    for (npy_intp r = 0; r < rb; ++r) {
        for (npy_intp c = 0; c < cb; ++c) {
            if (!M[r*sm0 + c*sm1]) continue;
            PyObject* acc = PyLong_FromLong(0);
            if (!acc) { Py_DECREF(rhs_mod); for (int j=0;j<16;++j) Py_DECREF(a_arr[j]); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
            int ok = 1;
            for (int k = 0; k < 16; ++k) {
                long long xv = X[r*sx0 + k*sx1] + 2;
                long long yv = Y[c*sy0 + k*sy1] + 2;
                unsigned long long c64 = (unsigned long long)xv * (unsigned long long)yv;
                PyObject* cPy = PyLong_FromUnsignedLongLong(c64);
                if (!cPy) { ok = 0; break; }
                PyObject* prod = PyNumber_Multiply(a_arr[k], cPy);
                Py_DECREF(cPy);
                if (!prod) { ok = 0; break; }
                PyObject* prodm = PyNumber_Remainder(prod, kobj);
                Py_DECREF(prod);
                if (!prodm) { ok = 0; break; }
                PyObject* s = PyNumber_Add(acc, prodm);
                Py_DECREF(prodm);
                if (!s) { ok = 0; break; }
                Py_DECREF(acc);
                acc = PyNumber_Remainder(s, kobj);
                Py_DECREF(s);
                if (!acc) { ok = 0; break; }
            }
            if (!ok) { Py_XDECREF(acc); Py_DECREF(rhs_mod); for (int j=0;j<16;++j) Py_DECREF(a_arr[j]); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
            int eq = PyObject_RichCompareBool(acc, rhs_mod, Py_EQ);
            Py_DECREF(acc);
            if (eq == 1) {
                long long sx = XS[r];
                long long sy = YS[c];
                Py_DECREF(rhs_mod); for (int j=0;j<16;++j) Py_DECREF(a_arr[j]); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask);
                return Py_BuildValue("LL", sx, sy);
            } else if (eq < 0) {
                Py_DECREF(rhs_mod); for (int j=0;j<16;++j) Py_DECREF(a_arr[j]); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL;
            }
        }
    }
    Py_DECREF(rhs_mod); for (int j=0;j<16;++j) Py_DECREF(a_arr[j]); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask);
    Py_RETURN_NONE;
}

// Helper: convert PyLong (decimal via PyObject_Str) to mpz_t
static int pylong_to_mpz(PyObject* obj, mpz_t z) {
    PyObject* s = PyObject_Str(obj);
    if (!s) return 0;
    const char* dec = PyUnicode_AsUTF8(s);
    if (!dec) { Py_DECREF(s); return 0; }
    if (mpz_set_str(z, dec, 10) != 0) { Py_DECREF(s); return 0; }
    Py_DECREF(s);
    return 1;
}

static PyObject* verify_first_match_mt(PyObject* self, PyObject* args) {
    PyObject *xobj, *yobj, *xseeds_obj, *yseeds_obj, *alist_obj, *kobj, *rhsobj, *mask_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &xobj, &yobj, &xseeds_obj, &yseeds_obj, &alist_obj, &kobj, &rhsobj, &mask_obj)) return NULL;
    PyArrayObject *xarr = (PyArrayObject*)PyArray_FROM_OTF(xobj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yarr = (PyArrayObject*)PyArray_FROM_OTF(yobj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *xseeds = (PyArrayObject*)PyArray_FROM_OTF(xseeds_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *yseeds = (PyArrayObject*)PyArray_FROM_OTF(yseeds_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *mask = (PyArrayObject*)PyArray_FROM_OTF(mask_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
    if (!xarr || !yarr || !xseeds || !yseeds || !mask) { Py_XDECREF(xarr); Py_XDECREF(yarr); Py_XDECREF(xseeds); Py_XDECREF(yseeds); Py_XDECREF(mask); return NULL; }
    if (!ensure_int64_2d(xarr, 16) || !ensure_int64_2d(yarr, 16) || !ensure_int64_1d(xseeds) || !ensure_int64_1d(yseeds)) {
        PyErr_SetString(PyExc_ValueError, "invalid array shapes/dtypes");
        Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL;
    }
    if (PyArray_NDIM(mask) != 2) { PyErr_SetString(PyExc_ValueError, "mask must be 2D bool array"); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
    npy_intp rb = PyArray_DIMS(xarr)[0];
    npy_intp cb = PyArray_DIMS(yarr)[0];
    if (PyArray_DIMS(mask)[0] != rb || PyArray_DIMS(mask)[1] != cb) { PyErr_SetString(PyExc_ValueError, "mask shape mismatch"); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
    if (PyArray_DIMS(xseeds)[0] != rb || PyArray_DIMS(yseeds)[0] != cb) { PyErr_SetString(PyExc_ValueError, "seed arrays length mismatch"); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
    if (!PySequence_Check(alist_obj) || PySequence_Size(alist_obj) != 16) { PyErr_SetString(PyExc_ValueError, "a_mod_k must be length 16"); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }

    // Prepare constants
    mpz_t k, rhs;
    mpz_init(k); mpz_init(rhs);
    if (!pylong_to_mpz(kobj, k) || !pylong_to_mpz(rhsobj, rhs)) { mpz_clear(k); mpz_clear(rhs); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
    mpz_t a16[16];
    for (int i=0;i<16;++i) { mpz_init(a16[i]); }
    for (int i=0;i<16;++i) {
        PyObject* it = PySequence_GetItem(alist_obj, i);
        if (!it) { for (int j=0;j<16;++j) mpz_clear(a16[j]); mpz_clear(k); mpz_clear(rhs); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
        if (!pylong_to_mpz(it, a16[i])) { Py_DECREF(it); for (int j=0;j<16;++j) mpz_clear(a16[j]); mpz_clear(k); mpz_clear(rhs); Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask); return NULL; }
        Py_DECREF(it);
        mpz_mod(a16[i], a16[i], k);
    }

    long long *X = (long long*)PyArray_DATA(xarr);
    long long *Y = (long long*)PyArray_DATA(yarr);
    long long *XS = (long long*)PyArray_DATA(xseeds);
    long long *YS = (long long*)PyArray_DATA(yseeds);
    npy_bool *M = (npy_bool*)PyArray_DATA(mask);
    npy_intp sx0 = PyArray_STRIDES(xarr)[0] / sizeof(long long);
    npy_intp sx1 = PyArray_STRIDES(xarr)[1] / sizeof(long long);
    npy_intp sy0 = PyArray_STRIDES(yarr)[0] / sizeof(long long);
    npy_intp sy1 = PyArray_STRIDES(yarr)[1] / sizeof(long long);
    npy_intp sm0 = PyArray_STRIDES(mask)[0] / sizeof(npy_bool);
    npy_intp sm1 = PyArray_STRIDES(mask)[1] / sizeof(npy_bool);

    volatile int found = 0;
    long long out_sx = 0, out_sy = 0;

    #pragma omp parallel
    {
        mpz_t acc, tmp;
        mpz_init(acc); mpz_init(tmp);
        #pragma omp for collapse(2)
        for (npy_intp r = 0; r < rb; ++r) {
            for (npy_intp c = 0; c < cb; ++c) {
                if (found) continue;
                if (!M[r*sm0 + c*sm1]) continue;
                mpz_set_ui(acc, 0);
                for (int kidx = 0; kidx < 16; ++kidx) {
                    unsigned long long xv = (unsigned long long)(X[r*sx0 + kidx*sx1] + 2);
                    unsigned long long yv = (unsigned long long)(Y[c*sy0 + kidx*sy1] + 2);
                    unsigned long long c64 = xv * yv;
                    mpz_mul_ui(tmp, a16[kidx], (unsigned long)c64);
                    mpz_add(acc, acc, tmp);
                    mpz_mod(acc, acc, k);
                }
                if (mpz_cmp(acc, rhs) == 0) {
                    #pragma omp critical
                    {
                        if (!found) {
                            found = 1; out_sx = XS[r]; out_sy = YS[c];
                        }
                    }
                }
            }
        }
        mpz_clear(acc); mpz_clear(tmp);
    }
    for (int i=0;i<16;++i) mpz_clear(a16[i]);
    mpz_clear(k); mpz_clear(rhs);
    Py_DECREF(xarr); Py_DECREF(yarr); Py_DECREF(xseeds); Py_DECREF(yseeds); Py_DECREF(mask);
    if (found) {
        return Py_BuildValue("LL", out_sx, out_sy);
    }
    Py_RETURN_NONE;
}

// Forward declare build_block_plus2 before method table
static PyObject* build_block_plus2(PyObject* self, PyObject* args);

static PyMethodDef Methods[] = {
    {"dot_mod", dot_mod, METH_VARARGS, "Compute (X * Y^T) mod p for int64 arrays of shape [rb,16] and [cb,16]."},
    {"dot_eq_mod_mask", dot_eq_mod_mask, METH_VARARGS, "Compute mask of (X * Y^T) % p == rhs for int64 arrays."},
    {"verify_first_match", verify_first_match, METH_VARARGS, "Find first (seedx,seedy) satisfying checksum over mask; returns (seedx,seedy) or None."},
    {"verify_first_match_mt", verify_first_match_mt, METH_VARARGS, "Multithreaded GMP: find first (seedx,seedy) satisfying checksum over mask."},
    {"build_block_plus2", (PyCFunction)build_block_plus2, METH_VARARGS, "Build contiguous int64 [n,16] from sequence of 16-long vectors, adding +2 to each element."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fastmod",
    "Fast modular dot products for 16-wide blocks",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_fastmod(void) {
    import_array();
    return PyModule_Create(&moduledef);
}

static PyObject* build_block_plus2(PyObject* self, PyObject* args) {
    PyObject* seq;
    if (!PyArg_ParseTuple(args, "O", &seq)) return NULL;
    PyObject* fast = PySequence_Fast(seq, "expected a sequence of vectors");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    npy_intp dims[2] = { (npy_intp)n, 16 };
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT64);
    if (!out) { Py_DECREF(fast); return NULL; }
    long long* O = (long long*)PyArray_DATA(out);
    for (Py_ssize_t r = 0; r < n; ++r) {
        PyObject* row = PySequence_Fast_GET_ITEM(fast, r);
        PyObject* row_fast = PySequence_Fast(row, "row must be a sequence");
        if (!row_fast) { Py_DECREF(fast); Py_DECREF(out); return NULL; }
        if (PySequence_Fast_GET_SIZE(row_fast) != 16) {
            PyErr_SetString(PyExc_ValueError, "each vector must have length 16");
            Py_DECREF(row_fast); Py_DECREF(fast); Py_DECREF(out); return NULL; }
        for (int k = 0; k < 16; ++k) {
            PyObject* it = PySequence_Fast_GET_ITEM(row_fast, k);
            long long v = PyLong_AsLongLong(it);
            if (PyErr_Occurred()) { Py_DECREF(row_fast); Py_DECREF(fast); Py_DECREF(out); return NULL; }
            O[r*16 + k] = v + 2;
        }
        Py_DECREF(row_fast);
    }
    Py_DECREF(fast);
    return (PyObject*)out;
}
