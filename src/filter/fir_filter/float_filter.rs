use super::super::super::group_delay::*;
use super::FIRFilter;

use num::complex::Complex;

pub trait Filter {
    type Float;
    type Complex;

    /// Computes the Complex Frequency response of the filter
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::{FIRFilter, float_filter::Filter};
    /// use solid::filter::firdes::*;
    /// use num::complex::Complex;
    /// 
    /// let coefs = match firdes_notch(25, 0.35, 120.0) {
    ///     Ok(coefs) => coefs,
    ///     _ => vec!()
    /// };
    /// let filter = FIRFilter::new(&coefs, 1.0);
    /// let response = Filter::frequency_response(&filter, 0.0);
    /// 
    /// assert_eq!(response.re.round(), 1.0);
    /// assert_eq!(response.im, 0.0);
    /// ```
    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex;

    /// Computes the group delay in samples
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::{FIRFilter, float_filter::Filter};
    /// use solid::filter::firdes;
    /// 
    /// let coefs = match firdes::firdes_notch(12, 0.35, 120.0) {
    ///     Ok(coefs) => coefs,
    ///     _ => vec!()
    /// };
    /// let filter = FIRFilter::new(&coefs, 1.0);
    /// let delay = Filter::group_delay(&filter, 0.0);
    /// 
    /// assert_eq!((delay + 0.5) as usize, 12);
    /// ```
    fn group_delay(&self, frequency: Self::Float) -> Self::Float;
}

impl Filter for FIRFilter<f32> {
    type Float = f32;
    type Complex = Complex<f32>;

    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex {
        let mut output: Complex<f32> = Complex::new(0.0, 0.0);

        for i in 0..self.coefs.len() {
            output += self.coefs[i] * (Complex::new(0.0, 1.0) * 2.0 * std::f32::consts::PI * frequency * i as f32).exp()
        }
        output *= self.scale;
        output
    }

    fn group_delay(&self, frequency: Self::Float) -> Self::Float {
        match fir_group_delay_f32(&self.coefs, frequency) {
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

impl Filter for FIRFilter<f64> {
    type Float = f64;
    type Complex = Complex<f64>;

    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex {
        let mut output: Complex<f64> = Complex::new(0.0, 0.0);

        for i in 0..self.coefs.len() {
            output += self.coefs[i] * (Complex::new(0.0, 1.0) * 2.0 * std::f64::consts::PI * frequency * i as f64).exp()
        }
        output *= self.scale;
        output
    }

    fn group_delay(&self, frequency: Self::Float) -> Self::Float {
        match fir_group_delay(&self.coefs, frequency) {
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