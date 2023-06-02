use std::fmt::Debug;

use super::*;

#[derive(Debug, Clone)]
pub struct InterpolatingIIRFilter<Coef, In> {
    filter: IIRFilter<Coef, In>,
    interpolation: usize,
}

impl<Coef: Copy + Num + Sum, In: Copy> InterpolatingIIRFilter<Coef, In> {
    /// Constructs a new, [`InterpolatingIIRFilter<Coef, In>`]
    ///
    /// Uses the input which represents the discrete coefficients of type `Coef`
    /// to create the filter. Does work on type `In` elements. It also interpolates the
    /// signal by `n` samples.
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let filter = InterpolatingIIRFilter::<f64, Complex<f64>>::new(&coefficients.0, &coefficients.1, IIRFilterType::Normal, 2);
    /// ```
    pub fn new(feed_forward: &[Coef], feed_back: &[Coef], iirtype: IIRFilterType, interpolation: usize) -> Result<Self, Box<dyn Error>> {
        if feed_forward.is_empty() {
            return Err(Box::new(IIRError(IIRErrorCode::NumeratorLengthZero)));
        } else if feed_back.is_empty() {
            return Err(Box::new(IIRError(IIRErrorCode::DenominatorLengthZero)));
        }

        if interpolation < 1 {
            return Err(Box::new(IIRError(IIRErrorCode::InterpolationLessThanOne)));
        }
        Ok(InterpolatingIIRFilter {
            filter: IIRFilter::new(feed_forward, feed_back, iirtype)?,
            interpolation
        })
    }

    /// Gets the interpolation of the filter
    ///
    /// Returns a `usize` that is the `n` interpolation amount
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let filter = InterpolatingIIRFilter::<f64, Complex<f64>>::new(&coefficients.0, &coefficients.1, IIRFilterType::Normal, 2).unwrap();
    /// assert_eq!(filter.get_interpolation(), 2);
    /// ```
    #[inline(always)]
    pub fn get_interpolation(&self) -> usize {
        self.interpolation
    }

    /// Returns the Numerator Coefs that the second order filter is using
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let filter = InterpolatingIIRFilter::<f64, Complex<f64>>::new(&coefficients.0, &coefficients.1, IIRFilterType::Normal, 2).unwrap();
    ///
    /// let numerators = filter.numerator_coefs().to_vec();
    /// let orig_ratio = coefficients.0[0] / coefficients.0[1];
    /// let new_ratio = numerators[0] / numerators[1];
    ///
    /// assert_eq!(orig_ratio, new_ratio);
    /// ```
    #[inline(always)]
    pub fn numerator_coefs(&self) -> Vec<Coef> {
        self.filter.numerator_coefs()
    }

    /// Returns the Denominator Coefs that the second order filter is using
    ///
    /// Example
    
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let filter = InterpolatingIIRFilter::<f64, Complex<f64>>::new(&coefficients.0, &coefficients.1, IIRFilterType::Normal, 2).unwrap();
    ///
    /// let denominators = filter.denominator_coefs().to_vec();
    /// let orig_ratio = coefficients.1[0] / coefficients.1[1];
    /// let new_ratio = denominators[1] / denominators[0];
    ///
    /// assert!((orig_ratio - new_ratio).abs() < 0.00001);
    /// ```
    #[inline(always)]
    pub fn denominator_coefs(&self) -> Vec<Coef> {
        self.filter.denominator_coefs()
    }

    /// Returns all the Second Order Internal Filters
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let filter = InterpolatingIIRFilter::<f64, Complex<f64>>::new(&coefficients.0, &coefficients.1, IIRFilterType::SecondOrder, 2).unwrap();
    ///
    /// let filters = filter.second_order_filters();
    ///
    /// assert_eq!(filters.len(), 1);
    /// ```
    #[inline(always)]
    pub fn second_order_filters(&self)  -> &Vec<SecondOrderFilter<Coef, In>> {
        self.filter.second_order_filters()
    }

    /// Returns the IIR Type, Most times this should just be a second order filter
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let filter = InterpolatingIIRFilter::<f64, Complex<f64>>::new(&coefficients.0, &coefficients.1, IIRFilterType::SecondOrder, 2).unwrap();
    /// 
    /// assert_eq!(*filter.iir_type(), IIRFilterType::SecondOrder);
    /// ```
    #[inline(always)]
    pub fn iir_type(&self) -> &IIRFilterType {
        self.filter.iir_type()
    }
}

impl<Coef: Copy + Num + Sum, In: Copy + Num, Out> Filter<In, Out> for InterpolatingIIRFilter<Coef, In>
where 
    DotProduct<Coef>: Execute<In, Out>,
    Coef: Mul<Complex<f64>, Output = Complex<f64>> + Conj<Output = Coef> + Real<Output = Coef>,
    In: Sub<Out, Output = In>,
    Out: Sub<Out, Output = In>,
{
    /// Executes type `In` and returns interpolated `Out` 
    ///
    /// `Out` is whatever the data type results in the multiplication of `Coef` and `In` along the interpolation factor
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use solid::filter::Filter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut filter = InterpolatingIIRFilter::new(&coefficients.0, &coefficients.1, IIRFilterType::SecondOrder, 2).unwrap();
    ///
    /// let output = filter.execute(1f64);
    /// assert_eq!(output, vec![0.05816769596076701, 0.119535296293297]);
    ///
    /// ```
    fn execute(&mut self, sample: In) -> Vec<Out> {
        let mut out = self.filter.execute(sample);
        for _ in 1..self.interpolation {
            out.append(&mut self.filter.execute(In::zero()));
        }
        out
    }

    /// Executes a block of type `In` and returns interpolated `Out` 
    ///
    /// `Out` is whatever the data type results in the multiplication of `Coef` and `In` along the interpolation factor
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use solid::filter::Filter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let interpolation = 5;
    /// let mut filter = InterpolatingIIRFilter::new(&coefficients.0, &coefficients.1, IIRFilterType::SecondOrder, interpolation).unwrap();
    ///
    /// let input = [1.0, 0.0, 1.0, 0.0, 1.0];
    /// let output = filter.execute_block(&input);
    ///
    /// assert_eq!(output.len(), input.len() * interpolation);
    ///
    /// ```
    fn execute_block(&mut self, samples: &[In]) -> Vec<Out> {
        let mut block: Vec<Out> = vec![];
        for &sample in samples.iter() {
            block.append(&mut self.execute(sample));
        }
        block
    }

    /// Computes the Complex Frequency response of the filter
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use solid::filter::Filter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut filter = InterpolatingIIRFilter::new(&coefficients.0, &coefficients.1, IIRFilterType::SecondOrder, 2).unwrap();
    /// let output = filter.execute_block(&[1.0, 0.0, 1.0, 0.0, 1.0]);
    /// 
    /// let freq_res = filter.frequency_response(0.0);
    /// 
    /// assert_eq!(freq_res, Complex::new(0.0, 0.0));
    /// ```
    fn frequency_response(&self, frequency: f64) -> Complex<f64> {
        self.filter.frequency_response(frequency)
    }
    
    /// Computes the group delay in samples
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iir::interp::InterpolatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use solid::filter::Filter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut filter = InterpolatingIIRFilter::new(&coefficients.0, &coefficients.1, IIRFilterType::SecondOrder, 2).unwrap();
    /// let output = filter.execute_block(&[1.0, 0.0, 1.0, 0.0, 1.0]);

    /// let delay = filter.group_delay(0.0);
    /// 
    /// assert_eq!(delay, 19.6774211296624);
    /// ```
    fn group_delay(&self, frequency: f64) -> f64 {
        self.filter.group_delay(frequency)
    }
}

impl<C: fmt::Display, T: fmt::Display> fmt::Display for InterpolatingIIRFilter<C, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "interpating IIR<{}: {}>", std::any::type_name::<C>(), self.interpolation)
    }
}