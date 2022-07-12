use std::ops::Neg;

use num::{Complex, Num};

pub trait Conj {
    type Output;

    fn conj(&self) -> Self::Output;
}

impl Conj for f32 {
    type Output = f32;

    #[inline(always)]
    fn conj(&self) -> Self::Output { self.clone() }
}

impl Conj for f64 {
    type Output = f64;

    #[inline(always)]
    fn conj(&self) -> Self::Output { self.clone() }
}

impl<T: Copy + Num + Neg<Output=T>> Conj for Complex<T> {
    type Output = Complex<T>;

    #[inline(always)]
    fn conj(&self) -> Self::Output {
        Complex::conj(self)
    }
}

pub trait Real {
    type Output;

    fn real(&self) -> Self::Output;
}

impl Real for f32 {
    type Output = f32;

    #[inline(always)]
    fn real(&self) -> Self::Output { self.clone() }
}

impl Real for f64 {
    type Output = f64;

    #[inline(always)]
    fn real(&self) -> Self::Output { self.clone() }
}

impl<T: Clone> Real for Complex<T> {
    type Output = T;

    #[inline(always)]
    fn real(&self) -> Self::Output { self.re.clone() }
}

pub trait Imag {
    type Output;

    fn imag(&self) -> Self::Output;
}

impl Imag for f32 {
    type Output = f32;

    #[inline(always)]
    fn imag(&self) -> Self::Output { 0.0 }
}

impl Imag for f64 {
    type Output = f64;

    #[inline(always)]
    fn imag(&self) -> Self::Output { 0.0 }
}

impl<T: Clone> Imag for Complex<T> {
    type Output = T;

    #[inline(always)]
    fn imag(&self) -> Self::Output { self.im.clone() }
}