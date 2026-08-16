//! Read-only NumPy views over immutable native plan metadata.

use numpy::ndarray::ArrayView1;
use numpy::{Element, PyArray1, PyArrayLike1, PyArrayMethods};
use pyo3::prelude::*;

/// Borrow native storage without copying and keep its Python owner alive.
pub(crate) fn readonly_view<'py, T: Element>(
    owner: Bound<'py, PyAny>,
    values: &[T],
) -> Bound<'py, PyArray1<T>> {
    let view = ArrayView1::from(values);
    // SAFETY: every caller passes a frozen Python owner whose backing vector is
    // never mutated or reallocated. NumPy retains `owner` as the base object.
    let array = unsafe { PyArray1::borrow_from_array(&view, owner) };
    drop(array.readwrite().make_nonwriteable());
    array
}

pub(crate) fn owned_vec<T: Copy + Element>(values: PyArrayLike1<'_, T>) -> Vec<T> {
    values.as_array().iter().copied().collect()
}
