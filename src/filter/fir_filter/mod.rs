//! A Finite Impulse Response Filter
//!
//! FIR Filters which are also known as non-recursive filters operate on discrete time samples.
//! The output _y_ is the convolution of the input *x* with the filter coefficients *coefs*.
//!
//! # Example
//!
//! ```
//! use solid::filter::fir_filter::FIRFilter;
//! use solid::filter::filter_traits::Filter;
//! use solid::filter::firdes;
//! use num::complex::Complex;
//!
//! let coefs = match firdes::firdes_notch(25, 0.35, 120.0) {
//!     Ok(coefs) => coefs,
//!     _ => vec!()
//! };
//! let filter = FIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0).unwrap();
//! ```

use crate::window;

use super::super::dot_product::{execute::Execute, Direction, DotProduct};
use super::super::resources::msb_index;
use super::super::window::Window;

use std::error::Error;
use std::{fmt, vec};
use std::iter::Sum;
use std::ops::Mul;

use num_traits::Num;

pub mod decimating_fir_filter;
pub mod interpolating_fir_filter;

#[derive(Debug)]
pub enum FIRErrorCode {
    CoefficientsLengthZero,
    DecimationLessThanOne,
    InterpolationLessThanOne,
    NotEnoughFilters,
}

#[derive(Debug)]
pub struct FIRError(pub FIRErrorCode);

impl fmt::Display for FIRError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FIR Filter Error {:?}", self.0)
    }
}

impl Error for FIRError {}

#[derive(Debug)]
#[allow(dead_code)]
pub struct FIRFilter<C, T> {
    scale: C,
    window: Window<T>,
    coefs: DotProduct<C>,
}

impl<C: Copy + Num + Sum, T: Copy> FIRFilter<C, T> {
    /// Constructs a new, `FIRFilter<C, T>`
    ///
    /// Uses the input which represents the discrete coefficients of type `C`
    /// to create the filter. Does work on type `T` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0).unwrap();
    /// ```
    pub fn new(coefficents: &[C], scale: C) -> Result<Self, Box<dyn Error>> {
        if coefficents.is_empty() {
            return Err(Box::new(FIRError(FIRErrorCode::CoefficientsLengthZero)));
        }
        Ok(FIRFilter {
            scale,
            window: Window::new(1 << msb_index(coefficents.len()), 0),
            coefs: DotProduct::new(coefficents, Direction::REVERSE),
        })
    }

    /// Sets the scale in which the output is multiplied
    ///
    /// Uses a input of `C` to modify the output scaling
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0).unwrap();
    /// filter.set_scale(2.0);
    ///
    /// assert_eq!(filter.get_scale(), 2.0);
    /// ```
    #[inline(always)]
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
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0).unwrap();
    /// assert_eq!(filter.get_scale(), 1.0);
    /// ```
    #[inline(always)]
    pub fn get_scale(&self) -> C {
        self.scale
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
    /// use num::complex::Complex;
    ///
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0).unwrap();
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0), Complex::new(1.02, 0.0),
    ///     Complex::new(0.23, 0.0), Complex::new(9.19, 0.0)];
    /// let mut output;
    /// for &val in window.iter() {
    ///     output = filter.execute(val);
    /// }
    ///
    /// assert_eq!(output, Complex::new(60.03, 0.0));
    /// ```
    #[inline(always)]
    pub fn execute<Out>(&mut self, sample: T) -> Out
    where
        DotProduct<C>: Execute<T, Output = Out>,
        Out: Mul<C, Output = Out>,
    {
        self.window.push(sample);
        Execute::execute(&self.coefs, &self.window.to_vec()) * self.scale
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
    /// use num::complex::Complex;
    ///
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0).unwrap();
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0), Complex::new(1.02, 0.0),
    ///     Complex::new(0.23, 0.0), Complex::new(9.19, 0.0)];
    /// let output = filter.execute_block(&window);
    ///
    /// assert_eq!(output[4], Complex::new(60.03, 0.0));
    /// ```
    #[inline(always)]
    pub fn execute_block<Out>(&mut self, samples: &[T]) -> Vec<Out>
    where
        DotProduct<C>: Execute<T, Output = Out>,
        Out: Mul<C, Output = Out>,
    {
        let mut block: Vec<Out> = vec![];
        for &sample in samples.iter() {
            block.push(self.execute(sample));
        }
        block
    }

    /// Gets the length of the coefficients
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use num::complex::Complex;
    /// let coefs = vec![0.0; 12];
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0).unwrap();
    /// let len = filter.len();
    ///
    /// assert_eq!(len, 12);
    /// ```
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.coefs.len()
    }

    /// Returns if the coefficients are empty
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use num::complex::Complex;
    /// let coefs = vec![0.0; 12];
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0).unwrap();
    /// assert_eq!(filter.is_empty(), false);
    /// ```
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.coefs.is_empty()
    }

    /// Gets a reference to the coefficients
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use num::complex::Complex;
    /// let coefs = vec![0.0; 12];
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0).unwrap();
    /// let ref_coefs = filter.coefficients();
    ///
    /// assert_eq!(coefs, *ref_coefs);
    /// ```
    #[inline(always)]
    pub fn coefficients(&self) -> &Vec<C> {
        self.coefs.coefficents()
    }
}

impl<C: fmt::Display, T: fmt::Display> fmt::Display for FIRFilter<C, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FIR<{}> [Scale={:.5}] [Coefficients={}]",
            std::any::type_name::<C>(),
            self.scale,
            self.coefs
        )
    }
}

#[derive(Debug)]
pub struct FIRFilterBank<C, T> {
    window: Window<T>,
    coefs: Vec<DotProduct<C>>,
}

impl<C: Copy + Num + Sum, T: Copy> FIRFilterBank<C, T> {
    /// Constructs a new, `FIRFilterBank<C, T>`
    ///
    /// Uses the input which represents the discrete coefficients of type `C`
    /// to create the filter. Does work on type `T` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir_filter::FIRFilterBank;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = FIRFilterBank::<f64, Complex<f64>>::new(&coefficients, 2).unwrap();
    /// ```
    pub fn new(coefficients: &[C], filters: usize) -> Result<Self, Box<dyn Error>> {
        if filters == 0 {
            return Err(Box::new(FIRError(FIRErrorCode::NotEnoughFilters)));
        } else if coefficients.is_empty() {
            return Err(Box::new(FIRError(FIRErrorCode::CoefficientsLengthZero)));
        }
    
        let mut coefs = vec![];
        let sub_len = coefficients.len() / filters;
    
        for filter in 0..filters {
            let mut rev_sub_coefs = Vec::with_capacity(sub_len);
            unsafe { rev_sub_coefs.set_len(sub_len) };
            for index in 0..sub_len {
                rev_sub_coefs[sub_len - index - 1] = coefficients[filter + index * filters];
            }
    
            coefs.push(DotProduct::new(&rev_sub_coefs, Direction::FORWARD));
        }

        Ok(FIRFilterBank { 
            window: Window::new(sub_len, 0), 
            coefs
        })
    }
}
