
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

use std::fmt;
use std::error::Error;

use num::complex::Complex;

#[derive(Debug)]
#[allow(dead_code)]
pub enum DelayErrorCode {
    EmptyCoefficients,
    FrequencyOutOfBounds
}

#[derive(Debug)]
pub struct DelayError(pub DelayErrorCode);

impl fmt::Display for DelayError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let error_name = match self.0 {
            DelayErrorCode::EmptyCoefficients => "Empty Coefficients",
            DelayErrorCode::FrequencyOutOfBounds => "Frequency Out of Bounds [-0.5, 0.5]"
        };
        write!(f, r#"Delay Error: {}"#, error_name)
    }
}

impl Error for DelayError {}

/// Function to calculate the FIR Group Delay
pub fn fir_group_delay(filter_coefficiencts: &[f64], frequency: f64) -> Result<f64, Box<dyn Error>> {
    if filter_coefficiencts.len() == 0 {
        return Err(Box::new(DelayError(DelayErrorCode::EmptyCoefficients)));
    } else if frequency < -0.5 || frequency > 0.5 {
        return Err(Box::new(DelayError(DelayErrorCode::FrequencyOutOfBounds)));
    }

    let mut t0: Complex<f64> = Complex::new(0.0, 0.0);
    let mut t1: Complex<f64> = Complex::new(0.0, 0.0);
    for i in 0..filter_coefficiencts.len() {
        t0 += filter_coefficiencts[i] * (Complex::new(0.0, 1.0) * 2.0 * std::f64::consts::PI * frequency * i as f64).exp() * i as f64;
        t1 += filter_coefficiencts[i] * (Complex::new(0.0, 1.0) * 2.0 * std::f64::consts::PI * frequency * i as f64).exp();
    }

    Ok((t0/t1).re)
}