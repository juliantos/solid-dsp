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
/// 
/// assert_eq!((freq * 10000.0).round() / 10000.0, 1.9626);
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

/// Compute Bilinear Z Transform using polynomial expansion in pole-zero form
/// 
/// # Arguments
/// 
/// * `analog_zeros` - Zeros of the analog s-domain transfer function
/// * `analog_poles` - Poles of the analog s-domain transfer function
/// * `nominal_gain` - Scalar gain of the s-domain transfer function
/// * `frequency_pre_warp` -  Scalar sample rate
/// 
/// # Examples
/// 
/// ```
/// use solid::filter::iirdes::*;
/// use num::complex::Complex;
/// 
/// let pre_warp = frequency_pre_warp(0.35, 0.0, BandType::LOWPASS);
/// let analog_zeros = vec![Complex::new(-0.1, 4.0), Complex::new(1.0, 0.1), Complex::new(3000.0, -2.0)];
/// let analog_poles = vec![Complex::new(0.0, 3.0), Complex::new(2.1, 3.1), Complex::new(0.0, 0.0)];
/// let gain = Complex::new(1.21, 0.717);
/// let (digital_zeros, digital_poles, digital_gain) = bilinear_analog_to_digital(&analog_zeros, &analog_poles, gain, pre_warp);
/// 
/// 
/// assert_eq!(digital_zeros.len(), 3);
/// assert_eq!(digital_poles.len(), 3);
/// assert_eq!(digital_gain, Complex::new(0.0, 0.0));
/// ```
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

/// Compute Bilinear Z Transform using polynomial expansion specified by numerator and denominator
/// 
/// See reference: https://web.mit.edu/2.14/www/Handouts/PoleZero.pdf
/// 
/// # Arguments
/// 
/// * `numerators` - Numerator Coefficients of the analog transfer function
/// * `denominators` - Denominator Coefficients of the analog transfer function
/// * `bilateral_warping_factor` - Scalar for the `sample rate` warping factor
/// 
/// # Examples
/// 
/// ```
/// use solid::filter::iirdes::*;
/// use num::complex::Complex;
/// 
/// let pre_warp = frequency_pre_warp(0.35, 0.0, BandType::LOWPASS);
/// let numerators = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)];
/// let denominators = vec![Complex::new(0.0, 1.0), Complex::new(0.0, 2.0), Complex::new(0.0, 3.0)];
/// 
/// let (digital_zeros, digital_poles) = bilinear_numerator_denominator(&numerators, &denominators, pre_warp).unwrap_or((vec![], vec![]));
/// 
/// assert_eq!(digital_zeros, vec![Complex::new(0.0, -1.0), Complex::new(0.0, -1.0)]);
/// assert_eq!(digital_poles, vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)]);
/// ```
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


// TODO: Bilinear Space-State Transforms

/// Flips a digital low-pass filter to a digital high-pass filter or vice-versa
/// 
/// # Arguments
/// 
/// * `zeros` - Digital Zeros
/// * `poles` - Digital Poles
/// 
/// # Examples
/// 
/// ```
/// use solid::filter::iirdes::*;
/// use num::complex::Complex;
/// 
/// let zeros = [Complex::new(1.0, -1.0)];
/// let poles = [Complex::new(2.0, -2.0)];
/// 
/// let (new_zeros, new_poles) = digital_filter_flip_pass(&zeros, &poles).unwrap();
/// 
/// assert_eq!(new_zeros, [Complex::new(-1.0, 1.0)]);
/// assert_eq!(new_poles, [Complex::new(-2.0, 2.0)]);
/// ```
pub fn digital_filter_flip_pass(zeros: &[Complex<f64>], poles: &[Complex<f64>]) -> Result<(Vec<Complex<f64>>, Vec<Complex<f64>>), Box<dyn Error>> {
    if zeros.len() != poles.len() {
        return Err(Box::new(IirdesError(IirdesErrorCode::InvalidOrder)));
    }

    let output_zeros: Vec<Complex<f64>> = zeros.iter().map(|x| -x).collect();
    let output_poles: Vec<Complex<f64>> = poles.iter().map(|x| -x).collect();

    Ok((output_zeros, output_poles))
}

/// Flips a digital low-pass filter to a digital high-pass filter or vice-versa
/// 
/// # Arguments
/// 
/// * `zeros` - Digital Zeros
/// * `poles` - Digital Poles
/// * `shift` - Frequency to shift
/// 
/// # Examples
/// 
/// ```
/// use solid::filter::iirdes::*;
/// use num::complex::Complex;
/// 
/// let zeros = [Complex::new(0.9, 0.0), Complex::new(0.9, 0.0), Complex::new(0.3, 0.0), Complex::new(0.1, 0.0), Complex::new(-0.5, 0.0)];
/// let poles = [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
/// 
/// let (new_zeros, new_poles) = digital_filter_shift(&zeros, &poles, 0.5).unwrap();
/// 
/// assert_eq!(new_zeros, []);
/// ```
pub fn digital_filter_shift(zeros: &[Complex<f64>], poles: &[Complex<f64>], shift: f64) -> Result<(Vec<Complex<f64>>, Vec<Complex<f64>>), Box<dyn Error>> {
    if zeros.len() != poles.len() {
        return Err(Box::new(IirdesError(IirdesErrorCode::InvalidOrder)));
    }

    let c = (2.0 * std::f64::consts::PI * shift).cos();
    let one = Complex::new(1.0, 0.0);
    let mut output_zeros = vec![Complex::new(0.0, 0.0); zeros.len() * 2];
    let mut output_poles = vec![Complex::new(0.0, 0.0); zeros.len() * 2];
    for i in 0..zeros.len() {
        let t = zeros[i] + one;
        output_zeros[2 * i + 0] = 0.5 * (c * t + (c * c * t * t - 4.0 * zeros[i]).sqrt());
        output_zeros[2 * i + 1] = 0.5 * (c * t - (c * c * t * t - 4.0 * zeros[i]).sqrt());

        let t = poles[i] + one;
        output_poles[2 * i + 0] = 0.5 * (c * t + (c * c * t * t - 4.0 * poles[i]).sqrt());
        output_poles[2 * i + 1] = 0.5 * (c * t - (c * c * t * t - 4.0 * poles[i]).sqrt());
    }

    Ok((output_zeros, output_poles))
}