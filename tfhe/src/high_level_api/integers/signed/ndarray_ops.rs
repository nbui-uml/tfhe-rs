use crate::high_level_api::integers::FheIntId;
use crate::prelude::*;
use crate::FheInt;
use std::ops::*;
use std::borrow::Borrow;

extern crate num_traits;
use num_traits::{One, Zero};
extern crate ndarray;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data};
use ndarray::{Ix1, Ix2};

impl<Id> Zero for FheInt<Id>
where
    FheInt<Id>: FheTrivialEncrypt<i8>,
    Id: FheIntId,
{
    fn zero() -> Self {
        return FheTrivialEncrypt::encrypt_trivial(0i8);
    }

    fn set_zero(&mut self) {
        *self = Self::zero();
    }

    // This is bad practice
    fn is_zero(&self) -> bool {
        return false;
    }
}

impl<Id> One for FheInt<Id>
where
    FheInt<Id>: FheTrivialEncrypt<i8>,
    Id: FheIntId,
{
    fn one() -> Self {
        return FheTrivialEncrypt::encrypt_trivial(1i8);
    }

    fn set_one(&mut self) {
        *self = Self::one();
    }

    fn is_one(&self) -> bool {
        return true;
    }
}

pub trait FheLinalgScalar:
    'static
    + Sized
    + Clone
    + Add<Output = Self>
    + for <'a> AddAssign<&'a Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Zero
    + One
    + Borrow<Self>
{
}
impl<Id> FheLinalgScalar for FheInt<Id> where Id: FheIntId {}

pub trait Accumulate {
    type Output;
    fn accumulate(&self) -> Self::Output;
}

impl<A, S> Accumulate for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    A: FheLinalgScalar,
{
    type Output = A;
    fn accumulate(&self) -> Self::Output {
        let mut sum = A::zero();
        for i in self {
            sum += i;
        }
        return sum;
    }
}

pub trait InnerProduct<Rhs> {
    type Output;
    fn inner_product(&self, rhs: &Rhs) -> Self::Output;
}

// vector-vector implementation
impl<A, S, S2> InnerProduct<ArrayBase<S2, Ix1>> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: FheLinalgScalar + Mul<Output = A>,
{
    type Output = A;

    fn inner_product(&self, rhs: &ArrayBase<S2, Ix1>) -> Self::Output {
        let prod = self * rhs;
        return prod.accumulate();
    }
}

// vector-matrix implementation
impl<A, S, S2> InnerProduct<ArrayBase<S2, Ix1>> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: FheLinalgScalar + Mul<Output = A>,
{
    type Output = Array1<A>;
    fn inner_product(&self, rhs: &ArrayBase<S2, Ix1>) -> Self::Output {
        let prod = self
            .axis_iter(Axis(0))
            .map(|i| i.inner_product(rhs))
            .collect();
        return prod;
    }
}

// matrix-matrix implementation
// impl<A, S, S2> InnerProduct<ArrayBase<S2, Ix2>> for ArrayBase<S, Ix2>
// where
//     S: Data<Elem = A>,
//     S2: Data<Elem = A>,
//     A: FheLinalgScalar,
// {
//     type Output = Array2<A>;
//     fn inner_product(&self, rhs: &ArrayBase<S2, Ix2>) -> Self::Output {
//         // let flattened: Array1<T> = source.into_iter().flat_map(|row| row.to_vec()).collect();
//         // let height = flattened.len() / width;
//         // flattened.into_shape((width, height));
//         // let prod = rhs
//         //     .axis_iter(Axis(1))
//         //     .map(|i| self.inner_product(&i)).
//         let mut prod = Zip::from(self.rows()).and(rhs.columns()).map_collect(|r, c| r.inner_product(&c));
//         prod.to_shape((self.nrows(), rhs.ncols()));
//         return prod;
//     }
// }

pub trait OuterProduct<RHS> {
    type Output;
    fn outer_product(&self, rhs: &RHS) -> Self::Output;
}

impl<A, S, S2> OuterProduct<ArrayBase<S2, Ix1>> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: FheLinalgScalar,
{
    type Output = Array2<A>;
    fn outer_product(&self, rhs: &ArrayBase<S2, Ix1>) -> Self::Output {
        let lhs = self.to_shape((1, self.dim())).unwrap().to_owned();
        let out = ndarray::concatenate(Axis(0), &vec![lhs.view(); rhs.dim()]).unwrap();

        return &out.t() * rhs;
    }
}
