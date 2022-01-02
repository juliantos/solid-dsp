//! A Finite Impulse Response Filter
//! 
//! FIR Filters which are also known as non-recursive filters operate on discrete time samples.
//! The output _y_ is the convolution of the input *x* with the filter coefficients *coefs*.
//! 
//! # Example
//! 
//! ```
//! use solid::filter::fir_filter::{FIRFilter, float_filter::Filter};
//! use solid::filter::firdes;
//! 
//! let coefs = match firdes::firdes_notch(25, 0.35, 120.0) {
//!     Ok(coefs) => coefs,
//!     _ => vec!()
//! };
//! let filter = FIRFilter::new(&coefs, 1.0);
//! ```

pub mod float_filter;

use super::super::dot_product::{DotProduct, Direction};
use super::super::circular_buffer::CircularBuffer;

use std::fmt;
use std::error::Error;
use std::ops::{Mul};

#[derive(Debug)]
#[allow(dead_code)]
pub struct FIRFilter<C> {
    coefs: Vec<C>,
    scale: C,
    window: CircularBuffer<C>,
    dot_product: DotProduct<C>
}

impl<C: Clone + Copy + Mul + std::iter::Sum<<C as std::ops::Mul>::Output>> FIRFilter<C> {
    /// Constructs a new, `FIRFilter<C>`
    /// 
    /// Uses the input which represents the discrete coeficients of type `C` 
    /// to create the filter.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// ```
    pub fn new(coefficents: &[C], scale: C) -> Self {
        FIRFilter {
            coefs: coefficents.to_vec(),
            scale: scale,
            window: CircularBuffer::new(1 << msb_index(coefficents.len())),
            dot_product: DotProduct::new(coefficents.to_vec(), Direction::REVERSE)
        }
    }

    /// Sets the scale in which the output is multiplied
    /// 
    /// Uses a input of `C` to modify the output scaling
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// filter.set_scale(2.0);
    /// 
    /// assert_eq!(filter.get_scale(), 2.0);
    /// ````
    pub fn set_scale(&mut self, scale: C) {
        self.scale = scale;
    }

    /// Gets the current scale in which the output is multipled
    ///
    /// Returns a `f64` that is the current scaling factor
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// assert_eq!(filter.get_scale(), 1.0);
    /// ```
    pub fn get_scale(&self) -> C {
        self.scale
    }

    /// Pushes a sample _x_ onto the internal buffer of the filter object
    /// 
    /// Returns an [`super::super::circular_buffer::BufferError`] if the window buffer cannot be applied to.
    /// Otherwise returns nothing.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// filter.push(4.0);
    /// ```
    pub fn push(&mut self, sample: C) -> Result<(), Box<dyn Error>> {
        self.window.push(sample)
    }

    /// Writes to the sample onto the internal buffer of the filter object
    /// 
    /// Returns an [`super::super::circular_buffer::BufferError`] if the window buffer cannot be applied to.
    /// Otherwise returns nothing.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// let window = [2.02, 4.04];
    /// filter.write(window.as_ptr(), 2);
    /// ```
    pub fn write(&mut self, samples: *const C, size: usize) -> Result<(), Box<dyn Error>> {
        self.window.write(samples, size)
    }

    /// Computes the output sample
    /// 
    /// The output is the dot product between the internal coefficients and the internal buffer
    /// and multiplied by the scaling factor
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// let window = [2.02, 4.04, 1.02, 0.23, 9.19];
    /// filter.write(window.as_ptr(), 5);
    /// let output = filter.execute();
    /// 
    /// assert_eq!(output as i64, 38);
    /// ```
    pub fn execute(&self) -> C::Output {
        let product = self.dot_product.execute(&self.window.to_vec()) * self.scale;
        product
    }

    /// Computes a [`Vec<C>`] of output samples
    /// 
    /// The output is the dot product between the internal coefficients and the internal buffer
    /// and multiplied by the scaling factor
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// let window = [2.02, 4.04, 1.02, 0.23];
    /// filter.write(window.as_ptr(), 4);
    /// let output = match filter.execute_block(vec![9.19, 1.2, 3.02, 0.3, 90.0]) {
    ///     Ok(output) => output,
    ///     Err(_) => vec![0.0]
    /// };
    /// 
    /// assert_eq!(output[0] as i64, 38);
    /// ```
    pub fn execute_block(&mut self, samples: Vec<C>) -> Result<Vec<C::Output>, Box<dyn Error>> {
        let mut block: Vec<C::Output> = vec![];
        for i in 0..samples.len() {
            if self.window.is_full() {
                self.window.pop()?;
            }
            self.push(samples[i])?;
            block.push(self.execute());
        }
        Ok(block)
    }

    /// Gets the length of the coefficients
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// let coefs = vec![0.0; 12];
    /// let filter = FIRFilter::new(&coefs, 1.0);
    /// let len = filter.len();
    /// 
    /// assert_eq!(len, 12);
    /// ```
    pub fn len(&self) -> usize {
        self.coefs.len()
    }

    /// Gets a reference to the coefficients
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// let coefs = vec![0.0; 12];
    /// let filter = FIRFilter::new(&coefs, 1.0);
    /// let ref_coefs = filter.coefficients();
    /// 
    /// assert_eq!(coefs, *ref_coefs);
    /// ```
    pub fn coefficients(&self) -> &Vec<C> {
        &self.coefs
    }
}

impl<C: Clone + fmt::Debug + fmt::Display> fmt::Display for FIRFilter<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, 
r#"FIR Filter 
Filter Type: {}
Coefficients: {:.4?}
Window: {}
"#, 
            std::any::type_name::<C>(), self.coefs, self.window)
    }
}


/// Gets the leading index bit location
/// TODO: move this to own file
/// 
/// # Example
/// 
/// ```
/// use solid::filter::fir_filter::msb_index;
/// let value = 0b1;
/// let index = msb_index(value);
/// assert_eq!(index, 1);
/// 
/// let value = 129;
/// let index = msb_index(value);
/// assert_eq!(index, 8);
/// 
/// ```
pub fn msb_index(x: usize) -> usize {
    (usize::BITS - x.leading_zeros()) as usize
}