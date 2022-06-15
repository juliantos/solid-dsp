use super::DotProduct;

use std::iter::Sum;
use std::ops::Mul;

use num::complex::Complex;
use num_traits::Num;

pub trait Execute<T> {
    type Output;

    /// Computes the Dot Product of the Samples and the Coefficients in the DotProduct
    /// Object.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::dot_product::{DotProduct, Direction, execute::Execute};
    /// 
    /// let coefs = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let dp = DotProduct::new(&coefs, Direction::REVERSE);
    /// let mul = vec![1.0; 5];
    /// let exe = Execute::execute(&dp, &mul);
    /// 
    /// assert_eq!(exe, 15.0);
    /// ```
    fn execute(&self, samples: &[T]) -> Self::Output;
}

impl<T: Copy + Mul + Sum<<T as Mul>::Output>> Execute<T> for DotProduct<T> {
    type Output = T;

    #[inline]
    fn execute(&self, samples: &[T]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl<T: Copy + Num> Execute<Complex<T>> for DotProduct<T> {
    type Output = Complex<T>;

    #[inline]
    fn execute(&self, samples: &[Complex<T>]) -> Self::Output {
        let product: Self::Output = samples.iter().zip(self.coef.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl<T: Copy + Num> Execute<T> for DotProduct<Complex<T>> {
    type Output = Complex<T>;

    #[inline]
    fn execute(&self, samples: &[T]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}