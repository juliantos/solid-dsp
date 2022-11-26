//! An Infinite Impulse Response Filter
//!
//! # Example
//!
//! ```
//! use solid::filter::*;
//! use num::complex::Complex;
//!
//! let coefs = match iirdes::pll::active_lag(0.35, 1.0 / (2.0f64).sqrt(), 1000.0) {
//!     Ok(coefs) => coefs,
//!     _ => (vec![], vec![])
//! };
//!
//! let filter = iir::IIRFilter::<f64, Complex<f64>>::new(&coefs.0, &coefs.1, iir::IIRFilterType::Normal);
//! ```

pub mod sos;

use crate::math::complex::Real;

use super::super::dot_product::{execute::Execute, Direction, DotProduct};
use super::super::window::Window;
use super::super::filter::Filter;
use super::super::group_delay::iir_group_delay;
use super::super::math::complex::Conj;
use sos::*;

use std::error::Error;
use std::fmt;
use std::iter::Sum;
use std::ops::{Sub, Mul};

use num_traits::Num;
use num::{Complex, Zero};

use either::Either::*;

#[derive(Debug)]
pub enum IIRErrorCode {
    NumeratorLengthZero,
    DenominatorLengthZero,
    SecondOrderSectionSizeZero,
    SecondOrderSectionSizeMismatch,
    SecondOrderSectionSizeNotMultpleOf3,
}

#[derive(Debug)]
pub struct IIRError(pub IIRErrorCode);

impl fmt::Display for IIRError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IIR Filter Error {:?}", self.0)
    }
}

impl Error for IIRError {}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum IIRFilterType {
    Normal,
    SecondOrder,
}

#[derive(Debug, Clone)]
pub struct IIRFilter<Coef, In> {
    iirtype: IIRFilterType,
    buffer: Window<In>,
    numerator_coefs: DotProduct<Coef>,
    denominator_coefs: DotProduct<Coef>,
    second_order_sections: Vec<SecondOrderFilter<Coef, In>>,
}

impl<Coef: Copy + Num + Sum, In: Copy> IIRFilter<Coef, In> {
    /// Creates and IIR Filter with coeffients of type `Coef` and takes in data of type `In`
    ///
    /// It should be noted that `In` and `Coef` are both numerican and can be multiplied and added together.
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir::*;
    /// use num::complex::Complex;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::<f64, Complex<f64>>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    /// ```
    pub fn new(
        feed_forward: &[Coef],
        feed_back: &[Coef],
        iirtype: IIRFilterType,
    ) -> Result<Self, Box<dyn Error>> {
        match iirtype {
            IIRFilterType::Normal => {
                if feed_forward.is_empty() {
                    return Err(Box::new(IIRError(IIRErrorCode::NumeratorLengthZero)));
                } else if feed_back.is_empty() {
                    return Err(Box::new(IIRError(IIRErrorCode::DenominatorLengthZero)));
                }

                let window_length = if feed_back.len() > feed_forward.len() {
                    feed_back.len()
                } else {
                    feed_forward.len()
                };

                let buffer = Window::new(window_length, 0);

                let a0 = feed_back[0];
                let mut numerator = Vec::new();
                let mut denominator = Vec::new();
                for &i in feed_forward.iter() {
                    numerator.push(i / a0);
                }
                for &i in feed_back.iter() {
                    denominator.push(i / a0);
                }

                Ok(IIRFilter {
                    iirtype,
                    buffer,
                    numerator_coefs: DotProduct::new(&numerator, Direction::FORWARD),
                    denominator_coefs: DotProduct::new(&denominator[1..], Direction::FORWARD),
                    second_order_sections: Vec::new(),
                })
            }
            IIRFilterType::SecondOrder => {
                if feed_forward.len() != feed_back.len() {
                    return Err(Box::new(IIRError(
                        IIRErrorCode::SecondOrderSectionSizeMismatch,
                    )));
                } else if feed_forward.is_empty() {
                    return Err(Box::new(IIRError(IIRErrorCode::SecondOrderSectionSizeZero)));
                } else if feed_forward.len() % 3 != 0 {
                    return Err(Box::new(IIRError(
                        IIRErrorCode::SecondOrderSectionSizeNotMultpleOf3,
                    )));
                }

                let len = feed_forward.len() / 3;
                let buffer = Window::new(len * 2, 0);

                let mut second_order_vector = Vec::new();
                for i in 0..len {
                    second_order_vector.push(SecondOrderFilter::new(
                        &feed_forward[(3 * i)..(3 * i + 3)],
                        &feed_back[(3 * i)..(3 * i + 3)],
                    )?);
                }

                Ok(IIRFilter {
                    iirtype,
                    buffer,
                    numerator_coefs: DotProduct::new(feed_forward, Direction::FORWARD),
                    denominator_coefs: DotProduct::new(feed_back, Direction::FORWARD),
                    second_order_sections: second_order_vector,
                })
            }
        }
    }

    /// Returns the Numerator Coefs that the second order filter is using
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir::*;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::<f64, f64>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let numerators = iir_filter.numerator_coefs();
    ///
    /// assert_eq!(numerators.to_vec(),  filter.0);
    /// ```
    #[inline(always)]
    pub fn numerator_coefs(&self) -> Vec<Coef> {
        self.numerator_coefs.coefficents()
    }

    /// Returns the Denominator Coefs that the second order filter is using
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir::*;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::<f64, f64>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let denominators = iir_filter.denominator_coefs();
    ///
    /// assert_eq!(denominators.to_vec(),  filter.1);
    /// ```
    #[inline(always)]
    pub fn denominator_coefs(&self) -> Vec<Coef> {
        self.denominator_coefs.coefficents()
    }

    /// Returns all the Second Order Internal Filters
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir::*;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::<f64, f64>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let filters = iir_filter.second_order_filters();
    ///
    /// assert_eq!(filters.len(), 1);
    /// ```
    #[inline(always)]
    pub fn second_order_filters(&self) -> &Vec<SecondOrderFilter<Coef, In>> {
        &self.second_order_sections
    }

    /// Returns the IIR Type, Most times this should just be a second order filter
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir::*;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::<f64, f64>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    /// assert_eq!(*iir_filter.iir_type(), IIRFilterType::SecondOrder);
    /// ```
    #[inline(always)]
    pub fn iir_type(&self) -> &IIRFilterType {
        &self.iirtype
    }
}

impl<Coef: Copy + Num + Sum, In: Copy, Out> Filter<In, Out> for IIRFilter<Coef, In>
where 
    DotProduct<Coef>: Execute<In, Out>,
    Coef: Mul<Complex<f64>, Output = Complex<f64>> + Conj<Output = Coef> + Real<Output = Coef>,
    In: Sub<Out, Output = In>,
    Out: Sub<Out, Output = In>,
{
    /// Executes type `In` and returns the data type `Out`
    ///
    /// `Out` is whatever the data type results in the multiplication of `Coef` and `In`
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir::*;
    /// use solid::filter::Filter;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let output = iir_filter.execute(1f64);
    ///
    /// assert_eq!(output[0], 0.05816769596076701);
    ///
    /// ```
    fn execute(&mut self, input: In) -> Vec<Out> {
        match self.iirtype {
            IIRFilterType::Normal => {
                let buffer = self.buffer.to_vec();
                let denom_output = self.denominator_coefs.execute(&buffer[..(buffer.len() - 1)]);
                let mixed_output = input - denom_output;

                self.buffer.push(mixed_output);

                vec![self.numerator_coefs.execute(&self.buffer.to_vec())]
            }
            IIRFilterType::SecondOrder => {
                let mut int_output = self.second_order_sections[0].execute::<Out>(Left(input));
                for second_order_filter in self.second_order_sections[1..].iter_mut() {
                    int_output = second_order_filter.execute(Right(int_output));
                }
                vec![int_output]
            }
        }
    }

    /// Executes array of type `In` and returns an array of the data type `Out`
    ///
    /// `Out` is whatever the data type results in the multiplication of `Coef` and `In`
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir::*;
    /// use solid::filter::Filter;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let output = iir_filter.execute_block(&[1.0, 0.0, 1.0, 0.0, 1.0]);
    ///
    /// assert_eq!(output, [0.05816769596076701, 0.119535296293297, 0.18410279587774706, 0.2518701895942824, 0.32283747232307686]);
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
    /// use solid::filter::iirdes;
    /// use solid::filter::iir::*;
    /// use solid::filter::Filter;
    /// use num::complex::Complex;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    /// let output = iir_filter.execute_block(&[1.0, 0.0, 1.0, 0.0, 1.0]);

    /// let freq_res = iir_filter.frequency_response(0.0);
    /// 
    /// assert_eq!(freq_res, Complex::new(0.0, 0.0));
    /// ```
    fn frequency_response(&self, frequency: f64) -> Complex<f64> {
        match self.iir_type() {
            IIRFilterType::Normal => {
                let mut b: Complex<f64> = Complex::zero();
                let mut a: Complex<f64> = Complex::zero();

                for (i, &coef) in self.numerator_coefs().iter().enumerate() {
                    let exp: Complex<f64> = coef
                        * Complex::from_polar(
                            1.0,
                            frequency * 2.0 * std::f64::consts::PI * (i as f64),
                        );
                    b += exp;
                }

                for (i, &coef) in self.denominator_coefs().iter().enumerate() {
                    let exp: Complex<f64> = coef
                        * Complex::from_polar(
                            1.0,
                            frequency * 2.0 * std::f64::consts::PI * (i as f64),
                        );
                    a += exp;
                }

                b / a
            }
            IIRFilterType::SecondOrder => {
                let mut h = Complex::zero();

                for filter in self.second_order_filters() {
                    h *= filter.frequency_response(frequency);
                }

                h
            }
        }
    }

    /// Computes the group delay in samples
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir::*;
    /// use solid::filter::Filter;
    /// use num::complex::Complex;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    /// let output = iir_filter.execute_block(&[1.0, 0.0, 1.0, 0.0, 1.0]);

    /// let delay = iir_filter.group_delay(0.0);
    /// 
    /// assert_eq!(delay, 19.6774211296624);
    /// ```
    fn group_delay(&self, frequency: f64) -> f64 {
        match self.iir_type() {
            IIRFilterType::Normal => {
                match iir_group_delay(&self.numerator_coefs(), &self.denominator_coefs(), frequency) {
                    Ok(delay) => delay,
                    Err(e) => {
                        if cfg!(debug_assertions) {
                            println!("{}", e);
                        }
                        0.0
                    }
                }
            }
            IIRFilterType::SecondOrder => {
                let mut delay = 0.0;
                for filter in self.second_order_filters().iter() {
                    delay = delay + filter.group_delay(frequency) + 2.0;
                }
                delay
            }
        }
    }
}

impl<C: fmt::Display, T: fmt::Display> fmt::Display for IIRFilter<C, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IIR<{}>", std::any::type_name::<C>())
    }
}