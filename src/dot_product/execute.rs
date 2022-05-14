use super::DotProduct;

use std::iter::Sum;

use num::complex::Complex;
use num_traits::float::Float;

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
    /// let dp = DotProduct::new(coefs.to_vec(), Direction::REVERSE);
    /// let mul = vec![1.0; 5];
    /// let exe = Execute::execute(&dp, &mul);
    /// 
    /// assert_eq!(exe, 15.0);
    /// ```
    fn execute(&self, samples: &[T]) -> Self::Output;
}

impl<T: Float + Sum> Execute<T> for DotProduct<T> {
    type Output = T;

    #[inline]
    fn execute(&self, samples: &[T]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl<T: Float + Sum> Execute<Complex<T>> for DotProduct<T> {
    type Output = Complex<T>;

    #[inline]
    fn execute(&self, samples: &[Complex<T>]) -> Self::Output {
        let product_real = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y.re).sum();
        let product_imag = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y.im).sum();
        Complex::new(product_real, product_imag)
    }
}

impl<T: Float> Execute<T> for DotProduct<Complex<T>> {
    type Output = Complex<T>;

    #[inline]
    fn execute(&self, samples: &[T]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}