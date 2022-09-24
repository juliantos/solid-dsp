//! A Finite Impulse Response Filter
//! 
//! FIR Filters which are also known as non-recursive filters operate on discrete time samples.
//! The output _y_ is the convolution of the input *x* with the filter coefficients *coefs*.
//! 
//! # Example
//! 
//! ```
//! use solid::filter::fir_filter::FIRFilter;
//! use solid::filter::filter::Filter;
//! use solid::filter::firdes;
//! use num::complex::Complex;
//! 
//! let coefs = match firdes::firdes_notch(25, 0.35, 120.0) {
//!     Ok(coefs) => coefs,
//!     _ => vec!()
//! };
//! let filter = FIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0);
//! ```

use super::super::dot_product::{DotProduct, Direction, execute::Execute};
use super::super::window::Window;
use super::super::resources::msb_index;

use std::fmt;
use std::iter::Sum;
use std::ops::Mul;

use num_traits::Num;

#[derive(Debug)]
#[allow(dead_code)]
pub struct FIRFilter<C, T> {
    scale: C,
    window: Window<T>,
    coefs: DotProduct<C>
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
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0);
    /// ```
    pub fn new(coefficents: &[C], scale: C) -> Self {
        // FIXME[epic=Error] Add error if size of coeffs is 0
        FIRFilter {
            scale: scale,
            window: Window::new(1 << msb_index(coefficents.len()), 0),
            coefs: DotProduct::new(coefficents, Direction::REVERSE)
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
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0);
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
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0);
    /// assert_eq!(filter.get_scale(), 1.0);
    /// ```
    #[inline(always)]
    pub fn get_scale(&self) -> C {
        self.scale
    }

    /// Pushes a sample _x_ onto the internal buffer of the filter object
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0);
    /// filter.push(Complex::new(4.0, 0.0));
    /// ```
    #[inline(always)]
    pub fn push(&mut self, sample: T) {
        self.window.push(sample);
    }

    /// Writes the samples onto the internal buffer of the filter object
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0);
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0)];
    /// filter.write(&window);
    /// ```
    #[inline(always)]
    pub fn write(&mut self, samples: &[T]) {
        self.window.write(samples)
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
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0);
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0), Complex::new(1.02, 0.0),
    ///     Complex::new(0.23, 0.0), Complex::new(9.19, 0.0)];
    /// filter.write(&window);
    /// let output = filter.execute();
    /// 
    /// assert_eq!(output.re.round(), 60.0);
    /// ```
    #[inline(always)]
    pub fn execute<Out>(&self) -> Out
    where DotProduct<C>: Execute<T, Output=Out>,
          Out: Mul<C, Output=Out>
    {
        let product = Execute::execute(&self.coefs, &self.window.to_vec()) * self.scale;
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
    /// use num::complex::Complex;
    /// 
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0);
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0), 
    ///     Complex::new(1.02, 0.0), Complex::new(0.23, 0.0)];
    /// filter.write(&window);
    /// let output = filter.execute_block(&vec![Complex::new(9.19, 0.0), Complex::new(1.2, 0.0), 
    ///         Complex::new(3.02, 0.0), Complex::new(0.3, 0.0), Complex::new(90.0, 0.0)]);
    /// 
    /// assert_eq!(output[0], Complex::new(60.03, 0.0));
    /// ```
    #[inline(always)]
    pub fn execute_block<Out>(&mut self, samples: &[T]) -> Vec<Out> 
    where DotProduct<C>: Execute<T, Output=Out>,
          Out: Mul<C, Output=Out>
    {
        let mut block: Vec<Out> = vec![];
        for &sample in samples.iter() {
            self.push(sample);
            block.push(self.execute());
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
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0);
    /// let len = filter.len();
    /// 
    /// assert_eq!(len, 12);
    /// ```
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.coefs.len()
    }

    /// Gets a reference to the coefficients
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use num::complex::Complex;
    /// let coefs = vec![0.0; 12];
    /// let filter = FIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0);
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
        write!(f, "FIR<{}> [Scale={:.5}] [Coefficients={}]", 
            std::any::type_name::<C>(), self.scale, self.coefs)
    }
}
