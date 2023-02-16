use super::*;

#[derive(Debug, Clone)]
pub struct DecimatingIIRFilter<Coef, In> {
    filter: IIRFilter<Coef, In>,
    decimation: usize,
}

impl<Coef: Copy + Num + Sum, In: Copy> DecimatingIIRFilter<Coef, In> {
    /// Constructs a new, [`DecimatingIIRFilter<Coef, In>`]
    ///
    /// Uses the input which represents the discrete coefficients of type `Coef`
    /// to create the filter. Does work on type `In` elements. It also decimates the
    /// signal by 1 in `n` samples.
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::iir::decim::DecimatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let filter = DecimatingIIRFilter::<f64, Complex<f64>>::new(&coefficients.0, &coefficients.1, IIRFilterType::Normal, 2);
    /// ```
    pub fn new(feed_forward: &[Coef], feed_back: &[Coef], iirtype: IIRFilterType, decimation: usize) -> Result<Self, Box<dyn Error>> {
        if feed_forward.is_empty() {
            return Err(Box::new(IIRError(IIRErrorCode::NumeratorLengthZero)));
        } else if feed_back.is_empty() {
            return Err(Box::new(IIRError(IIRErrorCode::DenominatorLengthZero)));
        }

        if decimation < 1 {
            return Err(Box::new(IIRError(IIRErrorCode::DecimationLessThanOne)));
        }
        Ok(DecimatingIIRFilter {
            filter: IIRFilter::new(feed_forward, feed_back, iirtype)?,
            decimation
        })
    }

    /// Gets the decimation of the filter
    ///
    /// Returns a `usize` that is the 1 in `n` decimation amount
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::iir::decim::DecimatingIIRFilter;
    /// use solid::filter::iir::*;
    /// use solid::filter::iirdes;
    /// use num::complex::Complex;
    /// 
    /// let coefficients = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let filter = DecimatingIIRFilter::<f64, Complex<f64>>::new(&coefficients.0, &coefficients.1, IIRFilterType::Normal, 2).unwrap();
    /// assert_eq!(filter.get_decimation(), 2);
    /// ```
    #[inline(always)]
    pub fn get_decimation(&self) -> usize {
        self.decimation
    }
}