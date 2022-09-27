use std::vec;

use super::*;

pub struct DecimatingFIRFilter<C, T> {
    filter: FIRFilter<C, T>,
    current_item: usize,
    decimation: usize
}

impl<C: Copy + Num + Sum, T: Copy> DecimatingFIRFilter<C, T> {
    /// Constructs a new, [`DecimatingFIRFilter<C, T>`]
    /// 
    /// Uses the input which represents the discrete coefficients of type `C` 
    /// to create the filter. Does work on type `T` elements. It also decimates the 
    /// signal by 1 in `n` samples.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::DecimatingFIRFilter;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0, 2);
    /// ```
    pub fn new(coefficents: &[C], scale: C, decimation: usize) -> Result<Self, Box<dyn Error>> {
        if coefficents.len() < 1 {
            return Err(Box::new(FIRError(FIRErrorCode::CoefficientsLengthZero)))
        } else if decimation < 1 {
            return Err(Box::new(FIRError(FIRErrorCode::DecimationLessThanOne)))
        }
        Ok(DecimatingFIRFilter {
            filter: FIRFilter {
                scale,
                window: Window::new(1 << msb_index(coefficents.len()), 0),
                coefs: DotProduct::new(coefficents, Direction::REVERSE)
            },
            current_item: 0,
            decimation
        })
    }

    /// Sets the scale in which the output is multiplied
    /// 
    /// Uses a input of `C` to modify the output scaling
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::DecimatingFIRFilter;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0, 2).unwrap();
    /// filter.set_scale(2.0);
    /// 
    /// assert_eq!(filter.get_scale(), 2.0);
    /// ```
    #[inline(always)]
    pub fn set_scale(&mut self, scale: C) {
        self.filter.scale = scale;
    }

    /// Gets the current scale in which the output is multipled
    ///
    /// Returns a `f64` that is the current scaling factor
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::DecimatingFIRFilter;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0, 2).unwrap();
    /// assert_eq!(filter.get_scale(), 1.0);
    /// ```
    #[inline(always)]
    pub fn get_scale(&self) -> C {
        self.filter.scale
    }

    /// Gets the decimation of the filter
    /// 
    /// Returns a `usize` that is the 1 in `n` decimation amount
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::*;
    /// use num::complex::Complex;
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0, 2).unwrap();
    /// assert_eq!(filter.get_decimation(), 2);
    /// ```
    #[inline(always)]
    pub fn get_decimation(&self) -> usize {
        self.decimation
    }

    /// Pushes a sample _x_ onto the internal buffer of the filter object
    /// 
    /// Increments the internal counter by 1 
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::DecimatingFIRFilter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0, 2).unwrap();
    /// filter.push(Complex::new(4.0, 0.0));
    /// ```
    #[inline(always)]
    pub fn push(&mut self, sample: T) {
        self.current_item = (self.current_item + 1) % self.decimation;
        self.filter.window.push(sample);
    }

    /// Writes the samples onto the internal buffer of the filter object
    /// 
    /// Also increments the internal counter
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::DecimatingFIRFilter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0, 2).unwrap();
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0)];
    /// filter.write(&window);
    /// ```
    #[inline(always)]
    pub fn write(&mut self, samples: &[T]) {
        self.current_item = (self.current_item + samples.len()) % self.decimation;
        self.filter.window.write(samples)
    }

    /// Computes the output sample
    /// 
    /// The output is the dot product between the internal coefficients and the internal buffer
    /// and multiplied by the scaling factor. It also only returns Some when the internal counter
    /// is equal to the current decimation rating.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::DecimatingFIRFilter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0, 2).unwrap();
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0)];
    /// filter.push(window[0]);
    /// let first_output = filter.execute();
    /// filter.push(window[1]);
    /// let second_output = filter.execute();
    /// assert_eq!(first_output, None);
    /// assert_eq!(second_output, Some(Complex::new(28.28, 0.0)));
    /// ```
    pub fn execute<Out>(&self) -> Option<Out>
    where DotProduct<C>: Execute<T, Output=Out>,
          Out: Mul<C, Output=Out>
    {
        if self.current_item == 0 {
            return Some(Execute::execute(&self.filter.coefs, &self.filter.window.to_vec()) * self.filter.scale);
        }
        None
    }

    /// Computes a [`Vec<C>`] of output samples
    /// 
    /// The output is the dot product between the internal coefficients and the internal buffer
    /// and multiplied by the scaling factor
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::DecimatingFIRFilter;
    /// use num::complex::Complex;
    /// 
    /// let coefficients: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 1.0, 2).unwrap();
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0), 
    ///     Complex::new(1.02, 0.0), Complex::new(0.23, 0.0)];
    /// let output = filter.execute_block(&window);
    /// 
    /// assert_eq!(output, vec![Complex::new(28.28, 0.0), Complex::new(21.39, 0.0)]);
    /// ```
    pub fn execute_block<Out>(&mut self, samples: &[T]) -> Vec<Out>
    where DotProduct<C>: Execute<T, Output=Out>,
          Out: Mul<C, Output=Out>
    {
        let mut block: Vec<Out> = vec![];
        for &sample in samples.iter() {
            self.push(sample);
            match self.execute() {
                Some(val) => block.push(val),
                None => {}
            }
        }
        block
    }

    /// Gets the length of the coefficients
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::DecimatingFIRFilter;
    /// use num::complex::Complex;
    /// let coefs = vec![0.0; 12];
    /// let filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0, 2).unwrap();
    /// let len = filter.len();
    /// 
    /// assert_eq!(len, 12);
    /// ```
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.filter.coefs.len()
    }

    /// Gets a reference to the coefficients
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::fir_filter::decimating_fir_filter::DecimatingFIRFilter;
    /// use num::complex::Complex;
    /// let coefs = vec![0.0; 12];
    /// let filter = DecimatingFIRFilter::<f64, Complex<f64>>::new(&coefs, 1.0, 2).unwrap();
    /// let ref_coefs = filter.coefficients();
    /// 
    /// assert_eq!(coefs, *ref_coefs);
    /// ```
    #[inline(always)]
    pub fn coefficients(&self) -> &Vec<C> {
        self.filter.coefs.coefficents()
    }
}

impl<C: fmt::Display, T: fmt::Display> fmt::Display for DecimatingFIRFilter<C, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FIR<{}> [Scale={:.5}] [Coefficients={}] [Decimation={}/{}]", 
            std::any::type_name::<C>(), self.filter.scale, self.filter.coefs, self.current_item, self.decimation)
    }
}
