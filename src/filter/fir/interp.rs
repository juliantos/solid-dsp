use std::fmt::Debug;

use super::*;
use super::pfb::*;

#[derive(Debug)]
pub struct InterpolatingFIRFilter<C, T> {
    filterbank: PolyPhaseFilterBank<C, T>,
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
    /// use solid::filter::fir::interp::*;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = InterpolatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 4);
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
        // Fixme: Init with zeros
        let effective_length = subfilter_len * interpolation;
        let mut effective_coefs: Vec<C> = Vec::with_capacity(effective_length);
        unsafe { effective_coefs.set_len(coefficents.len()) };
        effective_coefs.copy_from_slice(coefficents);
        unsafe { effective_coefs.set_len(effective_length) };

        let filterbank = PolyPhaseFilterBank::<C, T>::new(&effective_coefs, interpolation, C::one())?;

        Ok(InterpolatingFIRFilter {
            filterbank,
            interpolation
        })
    }
    
    #[inline(always)]
    pub fn set_scale(&mut self, scale: C) {
        self.filterbank.set_scale(scale)
    }

    #[inline(always)]
    pub fn get_scale(&self) -> C {
        self.filterbank.get_scale()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.filterbank.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.filterbank.is_empty()
    }

    #[inline(always)]
    pub fn coefficents(&self) -> Vec<Vec<C>> {
        self.filterbank.coefficents()
    }

    #[inline(always)]
    pub fn interpolation(&self) -> usize {
        self.interpolation
    }
}

impl<C: Copy + Num + Sum, T: Copy, Out> Filter<C, T, Out> for InterpolatingFIRFilter<C, T> 
where
    DotProduct<C>: Execute<T, Output = Out>
{
    fn execute(&mut self, sample: T) -> Vec<Out> {
        let mut interp_samples = vec![];
        self.filterbank.push(sample);
        for i in 0..self.filterbank.len() {
            interp_samples.push(self.filterbank.execute(i));
        }
        interp_samples
    }

    fn execute_block(&mut self, samples: &[T]) -> Vec<Out> {
        let mut interp_samples = vec![];
        samples.iter().for_each(|&sample| {
            self.filterbank.push(sample);
            for i in 0..self.filterbank.len() {
                interp_samples.push(self.filterbank.execute(i))
            }
        });
        interp_samples
    }

    fn frequency_response(&self, _frequency: f64) -> Complex<f64> {
        todo!("No Frequency Response for Interpolating Fir Filter")
    }

    fn group_delay(&self, _frequency: f64) -> f64 {
        todo!("No Group Delay for Interpolating Fir Filter")
    }
}