//! Generic Dot Product
//! 
//! Used to store coefficients and then to use them for computing the dot product against other vectors
//! 
//! # Example
//! 
//! ```
//! use solid::dot_product::{DotProduct, Direction};
//! 
//! let coefs = [1.0, 2.0, 3.0, 4.0, 5.0];
//! let dp = DotProduct::new(&coefs, Direction::REVERSE);
//! let mul = vec![1.0; 5];
//! let exe = dp.execute(&mul);
//! 
//! assert_eq!(exe, 15.0);
//! ```
pub mod execute;

use std::fmt;
use std::ops::Mul;
use std::iter::Sum;

pub enum Direction {
    FORWARD,
    REVERSE
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct DotProduct<T> {
    coef: Vec<T>
}

impl<T: Copy + Mul<T, Output=T> + Sum> DotProduct<T> {
    /// Creates a new Dot Product
    /// 
    /// Inserts the coefficients in the [`Direction`] specified. 
    /// 
    /// #Example
    /// 
    /// ```
    /// use solid::dot_product::{DotProduct, Direction};
    /// 
    /// let coefs = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let dp = DotProduct::new(&coefs, Direction::REVERSE);
    /// ```
    pub fn new(coefficients: &[T], direction: Direction) -> Self {
        let dot_product: DotProduct<T> = match direction {
            Direction::FORWARD => DotProduct { coef: coefficients.to_vec() },
            Direction::REVERSE => DotProduct {
                coef: coefficients.to_vec().into_iter().rev().collect()
            }
        };
        dot_product
    }

    /// Takes in a vector and applies the internal dot product
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::dot_product::{DotProduct, Direction};
    /// 
    /// let coefs = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let dp = DotProduct::new(&coefs, Direction::REVERSE);
    /// let mul = vec![1.0; 5];
    /// let exe = dp.execute(&mul);
    /// 
    /// assert_eq!(exe, 15.0);
    /// ```
    /// TODO[epic=fast]: SIMD, rebuild with different archs in mind
    #[inline(always)]
    pub fn execute(&self, samples: &[T]) -> T {
        let product: T = self.coef.iter().zip(samples.iter()).map(|(&x, &y)| x * y).sum();
        product
    }

}

impl<T: fmt::Debug> fmt::Display for DotProduct<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.coef)
    }
}