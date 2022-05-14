//! An Auto Correlation Filtering Object
//! 
//! The Auto Correlator is a combination of a delay, conjugate multiply and addition of the input signal.
//! 
//! # Example
//! 
//! ```
//! use solid::filter::auto_correlator::AutoCorrelator;
//! 
//! let auto_corr_filter = AutoCorrelator::<f64>::new(10, 5);
//! ```

use super::super::math::complex::{Real};
use super::super::window::Window;

// use std::fmt;
use std::error::Error;
use std::iter::Sum;

use num::complex::Complex;
use num_traits::{Num, Float};

#[derive(Debug)]
#[allow(dead_code)]
pub struct AutoCorrelator<C> {
    window_size: usize,
    delay: usize,
    current_delay: usize,
    window: Window<Complex<C>>,
    window_with_delay: Window<Complex<C>>,
    energy_buffer: Vec<f64>,
    energy_sum: f64,
    energy_index: usize
}

impl<C: Clone + Float + Sum + Num> AutoCorrelator<C> {
    /// Constructs a new, `AutoCorrelator<C>`
    /// 
    /// Creates an auto correlator filter of `window_size` and `delay` Also takes the primitive `C` 
    /// and makes it a `Complex<C>`. The last value of the initialization is the `energy_buffer`
    /// starting value should be 0.0.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::auto_correlator::AutoCorrelator;
    /// 
    /// let auto_corr_filter = AutoCorrelator::<f64>::new(10, 5);
    /// ```
    pub fn new(window_size: usize, delay: usize) -> Self {
        AutoCorrelator {
            window_size: window_size,
            delay: delay,
            current_delay: 0,
            window: Window::new(window_size, 0),
            window_with_delay: Window::new(window_size, delay),
            energy_buffer: vec![0.0; window_size],
            energy_sum: 0.0,
            energy_index: 0
        }
    }

    /// Resets the `AutoCorrelator` to its initial state
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::auto_correlator::AutoCorrelator;
    /// use num::Complex;
    /// 
    /// let mut auto_corr = AutoCorrelator::<f64>::new(10, 5);
    /// auto_corr.push(Complex::new(4.0, 0.0));
    /// auto_corr.reset();
    /// ```
    pub fn reset(&mut self) {
        self.window.reset();
        self.window_with_delay.reset();

        self.current_delay = 0;

        self.energy_sum = 0.0;
        self.energy_buffer = vec![0.0; self.window_size];
        self.energy_index = 0;
    }

    /// Pushes a sample _x_ onto both the internal buffers of the auto correlation object and updates
    /// the energy sum.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::auto_correlator::AutoCorrelator;
    /// use num::Complex;
    /// 
    /// let mut auto_corr = AutoCorrelator::<f64>::new(5, 10);
    /// auto_corr.push(Complex::new(4.0, 0.0));
    /// ```
    pub fn push(&mut self, sample: Complex<C>) 
    where Complex<C>: Real<Output=f64> {
        self.window.push(sample);
        self.window_with_delay.push(sample.conj());

        let e2 = (sample * sample.conj()).real();
        self.energy_sum = self.energy_sum - self.energy_buffer[self.energy_index];
        self.energy_sum = self.energy_sum + e2;
        self.energy_buffer[self.energy_index] = e2;
        self.energy_index = (self.energy_index + 1) % self.window_size;
    }

    /// Writes the samples onto the internal buffers of the auto correlator object
    /// 
    /// Returns an [`super::super::circular_buffer::BufferError`] if either window buffer cannot be applied to. 
    /// Otherwise return nothing.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::auto_correlator::AutoCorrelator;
    /// use num::Complex;
    /// 
    /// let mut auto_corr = AutoCorrelator::<f64>::new(5, 10);
    /// let window = [Complex::new(2.02, 0.0), Complex::new(4.04, 0.0)];
    /// auto_corr.write(&window);
    /// ```
    pub fn write(&mut self, samples: &[Complex<C>]) -> Result<(), Box<dyn Error>> 
    where Complex<C>: Real<Output=f64> {
        for &i in samples.iter() {
            self.push(i);
        }

        Ok(())
    }

    /// Uses the Two Internal Buffers to calculate the Auto Correlational Output
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::auto_correlator::AutoCorrelator;
    /// use num::Complex;
    /// 
    /// let len = 500;
    /// let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.05).collect();
    /// let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.05).collect();
    /// let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();
    /// 
    /// let mut auto_corr = AutoCorrelator::<f64>::new(5, 10);
    /// auto_corr.write(&complex_vec);
    /// let val = auto_corr.execute();
    /// ```
    pub fn execute(&self) -> Complex<C> {        
        self.window.to_vec().iter().zip(self.window_with_delay.to_vec().iter()).map(|(&x, &y)| {x * y}).sum()
    }

    /// Pushes Samples onto the buffer and then calculates the Auto Correlational Output
    /// 
    /// # Example 
    /// 
    /// ```
    /// use solid::filter::auto_correlator::AutoCorrelator;
    /// use num::Complex;
    /// 
    /// let len = 500;
    /// let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.05).collect();
    /// let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.05).collect();
    /// let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();
    /// 
    /// let mut auto_corr = AutoCorrelator::<f64>::new(5, 10);
    /// let output = auto_corr.execute_block(&complex_vec);
    /// ```
    pub fn execute_block(&mut self, samples: &[Complex<C>]) -> Vec<Complex<C>> 
    where Complex<C>: Real<Output=f64> {
        let mut block: Vec<Complex<C>> = vec![];
        for i in 0..samples.len() {
            self.push(samples[i]);
            block.push(self.execute());
        }
        block
    }

    /// Gets the Energy of the Auto Correlator
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::auto_correlator::AutoCorrelator;
    /// use num::Complex;
    /// 
    /// let len = 500;
    /// let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.05).collect();
    /// let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.05).collect();
    /// let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();
    /// 
    /// let mut auto_corr = AutoCorrelator::<f64>::new(5, 10);
    /// let output = auto_corr.execute_block(&complex_vec);
    /// let energy = auto_corr.get_energy();
    /// 
    /// assert_eq!((energy * 10000.0).round(), 125.0);
    /// ```
    pub fn get_energy(&self) -> f64 {
        self.energy_sum
    }
}