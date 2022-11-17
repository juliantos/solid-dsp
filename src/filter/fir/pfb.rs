use super::*;

#[derive(Debug)]
pub struct PolyPhaseFilterBank<C, T> {
    scale: C,
    window: Window<T>,
    coefs: Vec<DotProduct<C>>,
}

impl<C: Copy + Num + Sum, T: Copy> PolyPhaseFilterBank<C, T> {
    /// Constructs a new, `PolyPhaseFilterBank<C, T>`
    ///
    /// Uses the input which represents the discrete coefficients of type `C`
    /// to create the filter. Does work on type `T` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::pfb::PolyPhaseFilterBank;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = PolyPhaseFilterBank::<f64, Complex<f64>>::new(&coefficients, 2, 1.0).unwrap();
    /// ```
    pub fn new(coefficients: &[C], filters: usize, scale: C) -> Result<Self, Box<dyn Error>> {
        if filters == 0 {
            return Err(Box::new(FIRError(FIRErrorCode::NotEnoughFilters)));
        } else if coefficients.is_empty() {
            return Err(Box::new(FIRError(FIRErrorCode::CoefficientsLengthZero)));
        }
    
        let mut coefs = vec![];
        let sub_len = coefficients.len() / filters;
    
        for filter in 0..filters {
            let mut rev_sub_coefs = vec![C::zero(); sub_len];
            for index in 0..sub_len {
                rev_sub_coefs[sub_len - index - 1] = coefficients[filter + index * filters];
            }
    
            coefs.push(DotProduct::new(&rev_sub_coefs, Direction::FORWARD));
        }

        Ok(PolyPhaseFilterBank { 
            scale,
            window: Window::new(sub_len, 0), 
            coefs
        })
    }

    #[inline(always)]
    pub fn set_scale(&mut self, scale: C) {
        self.scale = scale
    }

    #[inline(always)]
    pub fn get_scale(&self) -> C {
        self.scale
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.coefs.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.coefs.is_empty()
    }

    #[inline(always)]
    pub fn coefficents(&self) -> Vec<Vec<C>> {
        self.coefs.iter().map(|coef| coef.coefficents().to_vec()).collect::<Vec<Vec<C>>>()
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.window.reset()
    }

    pub fn push(&mut self, sample: T) {
        self.window.push(sample)
    }

    pub fn execute<Out>(&mut self, index: usize) -> Out 
    where
        DotProduct<C>: Execute<T, Output = Out>
    {
        Execute::execute(&self.coefs[index], &self.window.to_vec())
    }
}
