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

use super::super::dot_product::{DotProduct, Direction, execute::Execute};
use super::super::window::Window;
use super::super::resources::msb_index;

use std::fmt;
use std::iter::Sum;

use num::complex::Complex;
use num_traits::{Num, Float};

#[derive(Debug)]
#[allow(dead_code)]
pub struct FIRFilter<C> {
    coefs: Vec<C>,
    scale: C,
    window: Window<Complex<C>>,
    dot_product: DotProduct<C>
}

impl<C: Clone + Float + Sum + Num> FIRFilter<C> {
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
            window: Window::new(1 << msb_index(coefficents.len()), 0),
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
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = FIRFilter::<f32>::new(&coefficients, 1.0);
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
    /// use num::Complex;
    /// 
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// filter.push(Complex::new(4.0, 0.0));
    /// ```
    #[inline(always)]
    pub fn push(&mut self, sample: Complex<C>) {
        self.window.push(sample);
    }

    /// Writes the samples onto the internal buffer of the filter object
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use num::Complex;
    /// 
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0)];
    /// filter.write(&window);
    /// ```
    #[inline(always)]
    pub fn write(&mut self, samples: &[Complex<C>]) {
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
    /// use num::Complex;
    /// 
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0), Complex::new(1.02, 0.0),
    ///     Complex::new(0.23, 0.0), Complex::new(9.19, 0.0)];
    /// filter.write(&window);
    /// let output = filter.execute();
    /// 
    /// assert_eq!(output.re.round(), 60.0);
    /// ```
    #[inline(always)]
    pub fn execute(&self) -> Complex<C> {
        let product = Execute::execute(&self.dot_product, &self.window.to_vec()) * self.scale;
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
    /// use num::Complex;
    /// 
    /// let coefficients: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = FIRFilter::<f32>::new(&coefficients, 1.0);
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0), 
    ///     Complex::new(1.02, 0.0), Complex::new(0.23, 0.0)];
    /// filter.write(&window);
    /// let output = filter.execute_block(&vec![Complex::new(9.19, 0.0), Complex::new(1.2, 0.0), 
    ///         Complex::new(3.02, 0.0), Complex::new(0.3, 0.0), Complex::new(90.0, 0.0)]);
    /// 
    /// assert_eq!(output[0], Complex::new(60.029995, 0.0));
    /// ```
    #[inline(always)]
    pub fn execute_block(&mut self, samples: &[Complex<C>]) -> Vec<Complex<C>> {
        let mut block: Vec<Complex<C>> = vec![];
        for i in 0..samples.len() {
            self.push(samples[i]);
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
    /// let coefs = vec![0.0; 12];
    /// let filter = FIRFilter::new(&coefs, 1.0);
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
    /// let coefs = vec![0.0; 12];
    /// let filter = FIRFilter::new(&coefs, 1.0);
    /// let ref_coefs = filter.coefficients();
    /// 
    /// assert_eq!(coefs, *ref_coefs);
    /// ```
    #[inline(always)]
    pub fn coefficients(&self) -> &Vec<C> {
        &self.coefs
    }
}

pub type FirFilter32 = FIRFilter<f32>;
pub type FirFilter64 = FIRFilter<f64>;

impl<C: Clone + fmt::Debug + fmt::Display + Copy> fmt::Display for FIRFilter<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, 
r#"FIR Filter 
Filter Type: {}
Coefficients: {:.4?}
Window: {:.4?}
"#, 
            std::any::type_name::<C>(), self.coefs, self.window.to_vec())
    }
}
