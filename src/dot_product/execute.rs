use super::DotProduct;

use num::complex::Complex;

pub trait Execute<T> {
    type Output;

    fn execute(&self, samples: &[T]) -> Self::Output;
}

impl Execute<f64> for DotProduct<f64> {
    type Output = f64;

    fn execute(&self, samples: &[f64]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl Execute<f64> for DotProduct<Complex<f64>> {
    type Output = Complex<f64>;

    fn execute(&self, samples: &[f64]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl Execute<Complex<f64>> for DotProduct<f64> {
    type Output = Complex<f64>;

    fn execute(&self, samples: &[Complex<f64>]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl Execute<Complex<f64>> for DotProduct<Complex<f64>> {
    type Output = Complex<f64>;

    fn execute(&self, samples: &[Complex<f64>]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl Execute<f32> for DotProduct<f32> {
    type Output = f32;

    fn execute(&self, samples: &[f32]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl Execute<f32> for DotProduct<Complex<f32>> {
    type Output = Complex<f32>;

    fn execute(&self, samples: &[f32]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl Execute<Complex<f32>> for DotProduct<f32> {
    type Output = Complex<f32>;

    fn execute(&self, samples: &[Complex<f32>]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}

impl Execute<Complex<f32>> for DotProduct<Complex<f32>> {
    type Output = Complex<f32>;

    fn execute(&self, samples: &[Complex<f32>]) -> Self::Output {
        let product: Self::Output = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }
}