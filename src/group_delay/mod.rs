
//! Functions for calculation both IIR and FIR Group Delay
//! Calculations use f64 as inputs
//! 
//! Example 
//! 
//! ```
//! use solid::group_delay::*;
//! 
//! let coefs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let delay = fir_group_delay(&coefs, 0.1);
//! ```

use super::math::complex::Conj;

use std::fmt;
use std::error::Error;

use std::ops::{Mul, Add};
use std::cmp::PartialOrd;

use num::complex::Complex;

static TOLERANCE: f64 = 0.00000000001;

#[derive(Debug)]
#[allow(dead_code)]
pub enum DelayErrorCode {
    EmptyCoefficients,
    FrequencyOutOfBounds,
    DivideByZero
}

#[derive(Debug)]
pub struct DelayError(pub DelayErrorCode);

impl fmt::Display for DelayError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let error_name = match self.0 {
            DelayErrorCode::EmptyCoefficients => "Empty Coefficients",
            DelayErrorCode::FrequencyOutOfBounds => "Frequency Out of Bounds [-0.5, 0.5]",
            DelayErrorCode::DivideByZero => "Denominator Coefficents Divide Numerator by Zero"
        };
        write!(f, r#"Delay Error: {}"#, error_name)
    }
}

impl Error for DelayError {}

/// Function to calculate the FIR Group Delay
pub fn fir_group_delay<C: Copy, T: Copy>(filter_coefficiencts: &[C], frequency: T) -> Result<f64, Box<dyn Error>> 
where T: PartialOrd<f64>,
      T: Mul<f64, Output=f64>,
      C: Mul<Complex<f64>, Output=Complex<f64>>
{
    if filter_coefficiencts.len() == 0 {
        return Err(Box::new(DelayError(DelayErrorCode::EmptyCoefficients)));
    } else if frequency < -0.5 || frequency > 0.5 {
        return Err(Box::new(DelayError(DelayErrorCode::FrequencyOutOfBounds)));
    }

    let mut t0: Complex<f64> = Complex::new(0.0, 0.0);
    let mut t1: Complex<f64> = Complex::new(0.0, 0.0);
    for i in 0..filter_coefficiencts.len() {
        let rot = Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
        t0 += filter_coefficiencts[i] * rot * (i as f64);
        t1 += filter_coefficiencts[i] * rot;
    }

    Ok((t0/t1).re)
}

/// Function to calculate the IIR Group Delay
pub fn iir_group_delay<C: Copy, T: Copy>(numerator_coefficients: &[C], denominator_coefficients: &[C], frequency: T) -> Result<f64, Box<dyn Error>>
where T: PartialOrd<f64>,
      T: Mul<f64, Output=f64>,
      C: Mul<Complex<f64>, Output=Complex<f64>>,
      C: Mul<C, Output=C>,
      C: Conj<Output=C>,
      C: Add<C, Output=C>
{
    if numerator_coefficients.len() == 0 {
        return Err(Box::new(DelayError(DelayErrorCode::EmptyCoefficients)));
    } else if denominator_coefficients.len() == 0 {
        return Err(Box::new(DelayError(DelayErrorCode::EmptyCoefficients)));
    } else if frequency < -0.5 || frequency > 0.5 {
        return Err(Box::new(DelayError(DelayErrorCode::FrequencyOutOfBounds)));
    }

    let coefs_len = numerator_coefficients.len() + denominator_coefficients.len() - 1;
    let mut coefs = Vec::<C>::with_capacity(coefs_len);
    unsafe { coefs.set_len(coefs_len) };

    for i in 0..denominator_coefficients.len() {
        for j in 0..numerator_coefficients.len() {
            let sum = denominator_coefficients[denominator_coefficients.len() - i - 1].conj() * numerator_coefficients[j];
            coefs[i + j] = coefs[i + j] + sum;
        }
    }

    let mut t0 = Complex::new(0.0, 0.0);
    let mut t1 = Complex::new(0.0, 0.0);
    for i in 0..coefs_len {
        let c0 = coefs[i] * Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
        t0 = t0 + (c0 * (i as f64));
        t1 = t1 + c0;
    }

    if t1.re.hypot(t1.im) <= TOLERANCE {
        return Err(Box::new(DelayError(DelayErrorCode::DivideByZero)));
    } 

    Ok((t0/t1).re - (denominator_coefficients.len() - 1) as f64)
}