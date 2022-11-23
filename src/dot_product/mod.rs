//! Generic Dot Product
//!
//! Used to store coefficients and then to use them for computing the dot product against other vectors
//!
//! # Example
//!
//! ```
//! use solid::dot_product::{DotProduct, Direction, execute::*};
//!
//! let coefs = [1.0, 2.0, 3.0, 4.0, 5.0];
//! let dp = DotProduct::new(&coefs, Direction::REVERSE);
//! let mul = vec![1.0; 5];
//! let exe = dp.execute(&mul);
//!
//! assert_eq!(exe, 15.0);
//! ```
pub mod execute;

extern crate alloc;

use std::alloc::Layout;
use std::cmp::{min};
use std::{fmt, mem, ptr};
use std::iter::Sum;
use std::ops::{Mul, AddAssign};

use num_traits::Num;

use self::execute::Execute;

pub enum Direction {
    FORWARD,
    REVERSE,
}

#[derive(Debug, Clone, Copy)]
pub struct DotProduct<T> {
    #[allow(dead_code)]
    layout: Layout,
    len: usize,
    buffer: *mut T
}

impl<T: Copy + Num + Sum> DotProduct<T> {
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
        let alignment = mem::align_of::<T>();
        let size = mem::size_of::<T>();
        let layout = match Layout::from_size_align(size * coefficients.len(), alignment) {
            Ok(layout) => layout,
            _ => panic!("Unable to create DotProduct of {}", coefficients.len())
        };
        let buffer = unsafe { alloc::alloc::alloc_zeroed(layout) } as *mut T;

        let dot_product: DotProduct<T> = match direction {
            Direction::FORWARD => {
                unsafe { std::ptr::copy_nonoverlapping(coefficients.as_ptr(), buffer, coefficients.len())}
                DotProduct {
                    layout,
                    len: coefficients.len(),
                    buffer
                }
            },
            Direction::REVERSE => {
                let mut rev_coefs = coefficients.to_vec();
                rev_coefs.reverse();
                unsafe { std::ptr::copy(rev_coefs.as_ptr(), buffer, coefficients.len()) };
                DotProduct {
                    layout,
                    len: coefficients.len(),
                    buffer
                }
            },
        };
        dot_product
    }

    /// Gets a reference to the coefficients
    ///
    /// # Example
    ///
    /// ```
    /// use solid::dot_product::{DotProduct, Direction};
    ///
    /// let coefs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let dp = DotProduct::new(&coefs, Direction::FORWARD);
    /// let ref_coefs = dp.coefficents();
    ///
    /// assert_eq!(coefs, ref_coefs);
    /// ```
    pub fn coefficents(&self) -> Vec<T> {
        let mut destination = Vec::with_capacity(self.len);
        unsafe {
            ptr::copy(self.buffer, destination.as_mut_ptr(), self.len);
            destination.set_len(self.len);
        }
        destination
    }

    /// Gets the length of the coefficients
    ///
    /// # Example
    ///
    /// ```
    /// use solid::dot_product::{DotProduct, Direction};
    ///
    /// let coefs = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let dp = DotProduct::new(&coefs, Direction::REVERSE);
    ///
    /// assert_eq!(dp.len(), 5);
    /// ```
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Gets if the coefficients are empty
    ///
    /// # Example
    ///
    /// ```
    /// use solid::dot_product::{DotProduct, Direction};
    ///
    /// let coefs = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let dp = DotProduct::new(&coefs, Direction::REVERSE);
    ///
    /// assert_eq!(dp.is_empty(), false);
    /// ```
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: fmt::Display> fmt::Display for DotProduct<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let typename = std::any::type_name::<T>();
        write!(f, "DotProduct<{}> [Size={}]", typename, self.len)
    }
}

impl<T: Copy, I: Copy, O: Num> Execute<I, O> for DotProduct<T> 
where 
    T: Mul<I, Output = O>,
    O: AddAssign
{
    #[inline]
    fn execute(&self, samples: &[I]) -> O {
        let iterations = min(samples.len(), self.len);
        let mut sum = O::zero();
        for (i, &sample) in samples.iter().enumerate().take(iterations) {
            let value = unsafe {
                let read_ptr = self.buffer.add(i);
                ptr::read(read_ptr)
            };
            sum += value * sample;
        }
        sum
    }
}

// impl<T> Drop for DotProduct<T> {
//     fn drop(&mut self) {
//         unsafe { alloc::alloc::dealloc(self.buffer as *mut u8, self.layout)}
//     }
// }