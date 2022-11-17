// TODO: Documentation, Group Delay. Print Statements
use super::super::super::dot_product::{execute::Execute, Direction, DotProduct};
use super::super::super::window::Window;
use super::iir_group_delay;

use crate::math::complex::*;

use std::error::Error;
use std::fmt;
use std::iter::Sum;
use std::ops::{Sub, Mul};

use either::Either;

use num::{Complex, Zero};
use num_traits::Num;

#[derive(Debug)]
pub enum SecondOrderErrorCode {
    CoefficientsNotInRange,
}

#[derive(Debug)]
pub struct SecondOrderError(pub SecondOrderErrorCode);

impl fmt::Display for SecondOrderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Second Order Error {:?}", self.0)
    }
}

impl Error for SecondOrderError {}

#[derive(Debug)]
pub struct SecondOrderFilter<C, T> {
    form_buffer_ii: Window<T>,
    numerator_coefs: DotProduct<C>,
    denominator_coefs: DotProduct<C>,
}

impl<C: Copy + Num + Sum, T: Copy> SecondOrderFilter<C, T> {
    /// Contructs a new `SecondOrderFilter<C, T>`
    ///
    /// Coeffients are of type C and the data being passed in and out is of type T.
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::iir::sos::SecondOrderFilter;
    /// use num::complex::Complex;
    ///
    /// let (ff_coefs, fb_coefs) = solid::filter::iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut second_order_filter = SecondOrderFilter::<f64, f64>::new(&ff_coefs, &fb_coefs).unwrap();
    /// ```
    pub fn new(feed_forward: &[C], feed_back: &[C]) -> Result<Self, Box<dyn Error>> {
        if feed_forward.len() < 3 || feed_back.len() < 3 {
            return Err(Box::new(SecondOrderError(
                SecondOrderErrorCode::CoefficientsNotInRange,
            )));
        }

        let a0 = feed_back[0];
        let b = [
            feed_forward[0] / a0,
            feed_forward[1] / a0,
            feed_forward[2] / a0,
        ];
        let a = [feed_back[0] / a0, feed_back[1] / a0, feed_back[2] / a0];

        Ok(SecondOrderFilter {
            form_buffer_ii: Window::new(3, 0),
            numerator_coefs: DotProduct::new(&a[1..], Direction::FORWARD),
            denominator_coefs: DotProduct::new(&b, Direction::FORWARD),
        })
    }

    /// Executes `T` or `Out` -> `Out` on a `SecondOrderFilter<C, T>`
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::iir::sos::SecondOrderFilter;
    /// use num::complex::Complex;
    ///
    /// let (ff_coefs, fb_coefs) = solid::filter::iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut second_order_filter = SecondOrderFilter::<f64, f64>::new(&ff_coefs, &fb_coefs).unwrap();
    ///
    /// let output = second_order_filter.execute(either::Either::Right(1.0));
    ///
    /// assert_eq!(output, 0.05816769596076701);
    /// ```
    pub fn execute<Out>(&mut self, input: Either<T, Out>) -> Out
    where
        DotProduct<C>: Execute<T, Output = Out>,
        T: Sub<Out, Output = T>,
        Out: Sub<Out, Output = T>,
    {
        let mut buffer = self.form_buffer_ii.to_vec();
        buffer[2] = buffer[1];
        buffer[1] = buffer[0];

        let denom_output = Execute::execute(&self.numerator_coefs, &buffer[1..]);

        let mixed_output = if input.is_left() {
            input.left().unwrap() - denom_output
        } else {
            input.right().unwrap() - denom_output
        };

        self.form_buffer_ii.push(mixed_output);
        let buffer = self.form_buffer_ii.to_vec();

        Execute::execute(&self.denominator_coefs, &buffer)
    }

    /// Returns the Numerator Coefs that the second order filter is using
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iir::sos::SecondOrderFilter;
    /// use num::complex::Complex;
    ///
    /// let (ff_coefs, fb_coefs) = solid::filter::iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let second_order_filter = SecondOrderFilter::<f64, f64>::new(&ff_coefs, &fb_coefs).unwrap();
    ///
    /// let numerators = second_order_filter.numerator_coefs();
    /// assert_eq!(numerators.len(), 2);
    /// assert_eq!(numerators[1], 0.99999840000128);
    /// ```
    #[inline(always)]
    pub fn numerator_coefs(&self) -> &Vec<C> {
        self.numerator_coefs.coefficents()
    }

    /// Returns the Denominator Coefs that the second order filter is using
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iir::sos::SecondOrderFilter;
    /// use num::complex::Complex;
    ///
    /// let (ff_coefs, fb_coefs) = solid::filter::iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let second_order_filter = SecondOrderFilter::<f64, f64>::new(&ff_coefs, &fb_coefs).unwrap();
    ///
    /// let denominators = second_order_filter.denominator_coefs();
    /// assert_eq!(denominators.len(), 3);
    /// assert_eq!(denominators[1], 0.003199997440002048);
    /// ```
    #[inline(always)]
    pub fn denominator_coefs(&self) -> &Vec<C> {
        self.denominator_coefs.coefficents()
    }

    /// Computes the Complex Frequency response of the filter
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::iir::sos::SecondOrderFilter;
    /// use num::complex::Complex;
    ///
    /// let (ff_coefs, fb_coefs) = solid::filter::iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let second_order_filter = SecondOrderFilter::<f64, f64>::new(&ff_coefs, &fb_coefs).unwrap();
    /// 
    /// let freq_res = second_order_filter.frequency_response(0.0);
    /// 
    /// assert!(freq_res != Complex::new(0.0, 0.0))
    /// ```
    pub fn frequency_response(&self, frequency: f64) -> Complex<f64> 
    where
        C: Mul<Complex<f64>, Output = Complex<f64>>
    {
        let mut output_b: Complex<f64> = Complex::zero();
        let mut output_a: Complex<f64> = Complex::zero();

        let coefs_b = self.numerator_coefs();
        let coefs_a = self.denominator_coefs();

        for (i, &coef) in coefs_b.iter().enumerate() {
            let polar = Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
            output_b = output_b + coef * polar;
        }
        for (i, &coef) in coefs_a.iter().enumerate() {
            let polar = Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
            output_a = output_a + coef * polar;
        }
        
        output_b / output_a
    }

    /// Computes the Group Delay in samples
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::filter::iir::sos::SecondOrderFilter;
    /// use num::complex::Complex;
    ///
    /// let (ff_coefs, fb_coefs) = solid::filter::iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let second_order_filter = SecondOrderFilter::<f64, f64>::new(&ff_coefs, &fb_coefs).unwrap();
    /// 
    /// let delay = second_order_filter.group_delay(0.0);
    /// 
    /// assert_eq!(delay, 17.6774211296624)
    /// ```
    pub fn group_delay(&self, frequency: f64) -> f64 
    where
        C: Real<Output = C> + Conj<Output = C> + Mul<Complex<f64>, Output = Complex<f64>>
    {
        let mut coefs_b = Vec::<C>::new();
        let mut coefs_a = Vec::<C>::new();

        for &b in self.numerator_coefs().iter() {
            coefs_b.push(b);
        }

        for &a in self.denominator_coefs().iter() {
            coefs_a.push(a);
        }

        match iir_group_delay(&coefs_b, &coefs_a, frequency) {
            Ok(delay) => delay + 2.0,
            Err(e) => {
                dbg!(e);
                0.0
            }
        }
    }
}
