//! Infinite Inpulse Response Filter Design

use super::super::math::poly;

use std::fmt;
use std::error::Error;

use num::complex::Complex;

pub enum BandType {
    LOWPASS,
    HIGHPASS,
    BANDPASS,
    BANDSTOP
}

#[derive(Debug)]
enum IirdesErrorCode {
    InvalidOrder,
    InvalidNumeratorSize
}

#[derive(Debug)]
struct IirdesError(IirdesErrorCode);

impl fmt::Display for IirdesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Iirdes Error: {:?}", self.0)
    }
}

impl Error for IirdesError {}

/// Computes the frequency pre-warping factor
/// 
/// # Arguments
/// 
/// * `cutoff` - The low-pass cutoff frequency
/// * `center_frequency` - The center frequency (bandpass/bandstop cases)
/// * `bandtype` - A filter design band type
/// 
/// # Example
/// 
/// ```
/// use solid::filter::iirdes::{frequency_pre_warp, BandType};
/// let freq = frequency_pre_warp(0.35, 0.0, BandType::LOWPASS);
/// ```
pub fn frequency_pre_warp(cutoff: f64, center_frequency: f64, bandtype: BandType) -> f64 {
    match bandtype {
        BandType::LOWPASS => (std::f64::consts::PI * cutoff).tan().abs(),
        BandType::HIGHPASS => { 
            let base = std::f64::consts::PI * cutoff; 
            (-(base.cos()) / base.sin()).abs() 
        }
        BandType::BANDPASS => {
            let base = 2.0 * std::f64::consts::PI * cutoff;
            let center = 2.0 * std::f64::consts::PI * center_frequency;
            ((base.cos() - center.cos()) / base.sin()).abs()
        }
        BandType::BANDSTOP => {
            let base = 2.0 * std::f64::consts::PI * cutoff;
            let center = 2.0 * std::f64::consts::PI * center_frequency;
            (base.sin() / (base.cos() - center.cos())).abs()
        }
    }
}

pub fn bilinear_analog_to_digital(analog_zeros: &[Complex<f64>], analog_poles: &[Complex<f64>], nominal_gain: Complex<f64>, frequency_pre_warp: f64) -> (Vec<Complex<f64>>, Vec<Complex<f64>>, Complex<f64>) {
    let mut digital_zeros: Vec<Complex<f64>> = vec![];
    let mut digital_poles: Vec<Complex<f64>> = vec![];
    
    let analog_zero_length = analog_zeros.len();
    let mut digital_gain = nominal_gain;
    for (i, &pole) in analog_poles.iter().enumerate() {
        let z;
        if i < analog_zero_length {
            let zm = analog_zeros[i] * frequency_pre_warp;
            z = (1.0 + zm) / (1.0 - zm);
        } else {
            z = Complex::new(-1.0, 0.0);
        }
        digital_zeros.push(z);

        let pm = pole * frequency_pre_warp;
        let p = (1.0 + pm) / (1.0 - pm);
        digital_poles.push(p);

        digital_gain *= (1.0 - p) / (1.0 - z);
    }

    (digital_zeros, digital_poles, digital_gain)
}

pub fn bilinear_numerator_denominator(numerators: &[Complex<f64>], denominators: &[Complex<f64>], bilateral_warping_factor: f64) -> Result<(Vec<Complex<f64>>, Vec<Complex<f64>>), Box<dyn Error>> {
    if numerators.len() == 0 || denominators.len() == 0 {
        return Err(Box::new(IirdesError(IirdesErrorCode::InvalidOrder)))
    }

    let numerator_order = numerators.len() - 1;
    let denominator_order = denominators.len() - 1;

    if numerator_order > denominator_order {
        return Err(Box::new(IirdesError(IirdesErrorCode::InvalidNumeratorSize)))
    }

    let mut numerator_output_digital_filter = Vec::with_capacity(numerator_order);
    let mut denominator_output_digital_filter = Vec::with_capacity(denominator_order);
    unsafe {
        numerator_output_digital_filter.set_len(numerator_order);
        denominator_output_digital_filter.set_len(denominator_order);
    }

    let mut mk = 1.0;
    let poly_1pz = poly::expand_binomial_pm(denominator_order, denominator_order - 1);
    for i in 0..denominator_order {
        for j in 0..denominator_order {
            denominator_output_digital_filter[j] = denominators[i] * mk * poly_1pz[j];
        }

        mk *= bilateral_warping_factor;
    }

    mk = 1.0;
    for i in 0..numerator_order {
        for j in 0..numerator_order {
            numerator_output_digital_filter[j] = numerators[i] * mk * poly_1pz[j];
        }

        mk *= bilateral_warping_factor;
    }

    let inverse_denominator_0 = 1.0 / denominator_output_digital_filter[0];
    for i in 0..denominator_order {
        denominator_output_digital_filter[i] *= inverse_denominator_0;
        numerator_output_digital_filter[i] *= inverse_denominator_0;
    }

    Ok((numerator_output_digital_filter, denominator_output_digital_filter))
}