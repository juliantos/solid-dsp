//! A Finite Impulse Response Filter
//!
//! FIR Filters which are also known as non-recursive filters operate on discrete time samples.
//! The output _y_ is the convolution of the input *x* with the filter coefficients *coefs*.
//!
//! # Example
//!
//! ```
//! use solid::filter::fir::FIRFilter;
//! use solid::filter::firdes;
//! use num::complex::Complex;
//!
//! let coefs = match firdes::firdes_notch(25, 0.35, 120.0) {
//!     Ok(coefs) => coefs,
//!     _ => vec!()
//! };
//! let filter = FIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0).unwrap();
//! ```


use super::super::dot_product::{execute::Execute, Direction, DotProduct};
use super::super::group_delay::fir_group_delay;
use super::super::resources::msb_index;
use super::super::window::Window;
use super::Filter;

use std::error::Error;
use std::{fmt, vec};
use std::iter::Sum;
use std::ops::Mul;

use num::{Complex, Zero};
use num_traits::Num;

pub mod decim;
pub mod interp;
pub mod pfb;

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

#[derive(Debug, Clone, Copy)]
pub struct FIRFilter<Coef, In> {
    scale: Coef,
    window: Window<In>,
    coefs: DotProduct<Coef>,
}

impl<Coef: Copy + Num + Sum, In: Copy> FIRFilter<Coef, In> {
    /// Constructs a new, `FIRFilter<Coef, In>`
    ///
    /// Uses the input which represents the discrete coefficients of type `Coef`
    /// to create the filter. Does work on type `In` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::FIRFilter;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0).unwrap();
    /// ```
    pub fn new(coefficents: &[Coef], scale: Coef) -> Result<Self, Box<dyn Error>> {
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
    /// Uses a input of `Coef` to modify the output scaling
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::FIRFilter;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0).unwrap();
    /// filter.set_scale(2.0);
    ///
    /// assert_eq!(filter.get_scale(), 2.0);
    /// ```
    #[inline(always)]
    pub fn set_scale(&mut self, scale: Coef) {
        self.scale = scale;
    }

    /// Gets the current scale in which the output is multipled
    ///
    /// Returns a `f64` that is the current scaling factor
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::FIRFilter;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0).unwrap();
    /// assert_eq!(filter.get_scale(), 1.0);
    /// ```
    #[inline(always)]
    pub fn get_scale(&self) -> Coef {
        self.scale
    }

    /// Gets the length of the coefficients
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::FIRFilter;
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
    /// use solid::filter::fir::FIRFilter;
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
    /// use solid::filter::fir::FIRFilter;
    /// use num::complex::Complex;
    /// let coefs = vec![0.0; 12];
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0).unwrap();
    /// let ref_coefs = filter.coefficients();
    ///
    /// assert_eq!(coefs, *ref_coefs);
    /// ```
    #[inline(always)]
    pub fn coefficients(&self) -> Vec<Coef> {
        self.coefs.coefficents()
    }
}

impl<Coef: Copy + Num + Sum, In: Copy, Out> Filter<In, Out> for FIRFilter<Coef, In>
where
    DotProduct<Coef>: Execute<In, Out>,
    Coef: Mul<Complex<f64>, Output = Complex<f64>>,
    Out: Mul<Coef, Output = Out>,
{

    /// Computes the output sample
    ///
    /// The output is the dot product between the internal coefficients and the internal buffer
    /// and multiplied by the scaling factor
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::FIRFilter;
    /// use solid::filter::Filter;
    /// use num::complex::Complex;
    ///
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0).unwrap();
    /// let window = vec![Complex::new(2.02, 0.0), Complex::new(4.04, 0.0), Complex::new(1.02, 0.0),
    ///     Complex::new(0.23, 0.0), Complex::new(9.19, 0.0)];
    /// let mut output = filter.execute(window[0]);
    ///
    /// assert_eq!(output[0], Complex::new(10.1, 0.0));
    /// ```
    #[inline(always)]
    fn execute(&mut self, sample: In) -> Vec<Out> {
        self.window.push(sample);
        vec![self.coefs.execute(&self.window.to_vec()) * self.scale]
    }

    /// Computes a [`Vec<C>`] of output samples
    ///
    /// The output is the dot product between the internal coefficients and the internal buffer
    /// and multiplied by the scaling factor
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::FIRFilter;
    /// use solid::filter::Filter;
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
    fn execute_block(&mut self, samples: &[In]) -> Vec<Out> {
        let mut block: Vec<Out> = vec![];
        for &sample in samples.iter() {
            block.append(&mut self.execute(sample));
        }
        block
    }

    /// Computes the Complex Frequency response of the filter
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::FIRFilter;
    /// use solid::filter::Filter;
    /// use solid::filter::firdes::*;
    /// use num::complex::Complex;
    ///
    /// let coefs = match firdes_notch(25, 0.35, 120.0) {
    ///     Ok(coefs) => coefs,
    ///     _ => vec!()
    /// };
    /// let filter = FIRFilter::<f64, f64>::new(&coefs, 1.0).unwrap();
    /// let response = Filter::frequency_response(&filter, 0.0);
    ///
    /// assert_eq!(response.re.round(), 1.0);
    /// assert_eq!(response.im, 0.0);
    /// ```
    fn frequency_response(&self, frequency: f64) -> Complex<f64> {
        let mut output: Complex<f64> = Complex::zero();

        let coefs = self.coefficients();
        for (i, coef) in coefs.iter().enumerate() {
            let out = *coef
                * Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
            output += out;
        }
        self.get_scale() * output
    }

    /// Computes the group delay in samples
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::FIRFilter;
    /// use solid::filter::Filter;
    /// use solid::filter::firdes;
    ///
    /// let coefs = match firdes::firdes_notch(12, 0.35, 120.0) {
    ///     Ok(coefs) => coefs,
    ///     _ => vec!()
    /// };
    /// let filter = FIRFilter::<f64, f64>::new(&coefs, 1.0).unwrap();
    /// let delay = Filter::group_delay(&filter, 0.0);
    ///
    /// assert_eq!((delay + 0.5) as usize, 12);
    /// ```
    fn group_delay(&self, frequency: f64) -> f64 {
        match fir_group_delay(&self.coefficients(), frequency) {
            Ok(delay) => delay,
            Err(e) => {
                if cfg!(debug_assertions) {
                    println!("{}", e);
                }
                0.0
            }
        }
    }
}

impl<C: fmt::Display, T: fmt::Display + Copy> fmt::Display for FIRFilter<C, T> {
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