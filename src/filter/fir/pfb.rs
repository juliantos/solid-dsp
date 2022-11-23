use super::*;

#[derive(Debug, Clone)]
pub struct PolyPhaseFilterBank<Coef, In> {
    scale: Coef,
    window: Window<In>,
    coefs: Vec<DotProduct<Coef>>,
}

impl<Coef: Copy + Num + Sum, In: Copy> PolyPhaseFilterBank<Coef, In> {
    /// Constructs a new, `PolyPhaseFilterBank<Coef, In>`
    ///
    /// Uses the input which represents the discrete coefficients of type `Coef`
    /// to create the filter. Does work on type `In` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir::pfb::PolyPhaseFilterBank;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = PolyPhaseFilterBank::<f64, Complex<f64>>::new(&coefficients, 2, 1.0).unwrap();
    /// ```
    pub fn new(coefficients: &[Coef], filters: usize, scale: Coef) -> Result<Self, Box<dyn Error>> {
        if filters == 0 {
            return Err(Box::new(FIRError(FIRErrorCode::NotEnoughFilters)));
        } else if coefficients.is_empty() {
            return Err(Box::new(FIRError(FIRErrorCode::CoefficientsLengthZero)));
        }
    
        let mut coefs = vec![];
        let sub_len = coefficients.len() / filters;
    
        for filter in 0..filters {
            let mut rev_sub_coefs = vec![Coef::zero(); sub_len];
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
    pub fn set_scale(&mut self, scale: Coef) {
        self.scale = scale
    }

    #[inline(always)]
    pub fn get_scale(&self) -> Coef {
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
    pub fn coefficents(&self) -> Vec<Vec<Coef>> {
        self.coefs.iter().map(|coef| coef.coefficents().to_vec()).collect::<Vec<Vec<Coef>>>()
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.window.reset()
    }

    pub fn push(&mut self, sample: In) {
        self.window.push(sample)
    }

    pub fn execute<Out>(&mut self, index: usize) -> Out 
    where
        DotProduct<Coef>: Execute<In, Out>
    {
        self.coefs[index].execute(&self.window.to_vec())
    }
}
