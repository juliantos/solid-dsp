use std::fmt::Debug;

use super::*;
use super::pfb::*;

#[derive(Debug, Clone)]
pub struct InterpolatingFIRFilter<Coef, In> {
    filterbank: PolyPhaseFilterBank<Coef, In>,
    interpolation: usize,
}

impl<Coef: Copy + Num + Sum + Debug, In: Copy> InterpolatingFIRFilter<Coef, In> {
    /// Constructs a new, [`InterpolatingFIRFilter<Coef, In>`]
    /// 
    /// Uses the input which represents discrete coefficients of the type `Coef`
    /// to create the filter banks. Does work on type `In` elements, It also iterpolates
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
    pub fn new(coefficents: &[Coef], interpolation: usize) -> Result<Self, Box<dyn Error>> {
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
        let mut effective_coefs: Vec<Coef> = vec![Coef::zero(); coefficents.len()];
        effective_coefs.copy_from_slice(coefficents);
        effective_coefs.resize(effective_length, Coef::zero());

        let filterbank = PolyPhaseFilterBank::<Coef, In>::new(&effective_coefs, interpolation, Coef::one())?;

        Ok(InterpolatingFIRFilter {
            filterbank,
            interpolation
        })
    }
    
    #[inline(always)]
    pub fn set_scale(&mut self, scale: Coef) {
        self.filterbank.set_scale(scale)
    }

    #[inline(always)]
    pub fn get_scale(&self) -> Coef {
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
    pub fn coefficents(&self) -> Vec<Coef> {
        self.filterbank.coefficents().iter().flat_map(|x| x.to_vec()).collect()
    }

    #[inline(always)]
    pub fn interpolation(&self) -> usize {
        self.interpolation
    }
}

impl<Coef: Copy + Num + Sum, In: Copy, Out> Filter<In, Out> for InterpolatingFIRFilter<Coef, In> 
where
    DotProduct<Coef>: Execute<In, Out>,
    Coef: Mul<Complex<f64>, Output = Complex<f64>>,
    Out: Mul<Coef, Output = Out>
{
    fn execute(&mut self, sample: In) -> Vec<Out> {
        let mut interp_samples = vec![];
        self.filterbank.push(sample);
        for i in 0..self.filterbank.len() {
            interp_samples.push(self.filterbank.execute(i));
        }
        interp_samples
    }

    fn execute_block(&mut self, samples: &[In]) -> Vec<Out> {
        let mut interp_samples = vec![];
        samples.iter().for_each(|&sample| {
            self.filterbank.push(sample);
            for i in 0..self.filterbank.len() {
                interp_samples.push(self.filterbank.execute(i))
            }
        });
        interp_samples
    }

    fn frequency_response(&self, frequency: f64) -> Complex<f64> {
        let mut output = Complex::zero();

        let coefs: Vec<Coef> = self.filterbank.coefficents().iter().flat_map(|x| x.to_vec()).collect();
        for (i, coef) in coefs.iter().enumerate() {
            let out = *coef
                * Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
            output += out;
        }

        self.filterbank.get_scale() * output
    }

    fn group_delay(&self, frequency: f64) -> f64 {
        let coefs: Vec<Coef> = self.filterbank.coefficents().iter().flat_map(|x| x.to_vec()).collect();
        match fir_group_delay(&coefs, frequency) {
            Ok(delay) => delay,
            Err(e) => {
                if cfg!(debug_assertions) {
                    dbg!(e);
                }
                0.0
            }
        }
    }
}