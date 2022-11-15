use std::fmt::Debug;

use super::*;

pub struct InterpolatingFIRFilter<C, T> {
    filterbank: FIRFilterBank<C, T>,
    interpolation: usize,
}

impl<C: Copy + Num + Sum + Debug, T: Copy + Debug> InterpolatingFIRFilter<C, T> {
    /// Constructs a new, [`InterpolatingFIRFilter<C, T>`]
    /// 
    /// Uses the input which represents discrete coefficients of the type 'C"
    /// to create the filter banks. Does work on type `T` elements, It also iterpolates
    /// the signal by `n` samples for each input sample.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::interpolating_fir_filter::*;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = InterpolatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0, 4);
    /// ```
    pub fn new(coefficents: &[C], interpolation: usize) -> Result<Self, Box<dyn Error>> {
        if coefficents.is_empty() {
            return Err(Box::new(FIRError(FIRErrorCode::CoefficientsLengthZero)));
        } else if interpolation < 1 {
            return Err(Box::new(FIRError(FIRErrorCode::InterpolationLessThanOne)));
        }

        // Subfilter length
        let coef_interp_length = coefficents.len() as f32 / interpolation as f32;
        let subfilter_len = if coef_interp_length == coef_interp_length.floor() {
            coef_interp_length as usize
        } else {
            coef_interp_length.ceil() as usize 
        };

        // Effective Filter
        let effective_length = subfilter_len * interpolation;
        let mut effective_coefs: Vec<C> = Vec::with_capacity(effective_length);
        unsafe { effective_coefs.set_len(coefficents.len()) };
        effective_coefs.copy_from_slice(coefficents);
        unsafe { effective_coefs.set_len(effective_length) };

        let filterbank = FIRFilterBank::<C, T>::new(&effective_coefs, interpolation)?;

        Ok(InterpolatingFIRFilter {
            filterbank,
            interpolation
        })
    }
}