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
//! let filter = iir_filter::IIRFilter::<f64, Complex<f64>>::new(&coefs.0, &coefs.1, iir_filter::IIRFilterType::Normal);
//! ```

pub mod second_order_filter;

use super::super::dot_product::{execute::Execute, Direction, DotProduct};
use super::super::window::Window;
use second_order_filter::SecondOrderFilter;

use std::error::Error;
use std::fmt;
use std::iter::Sum;
use std::ops::Sub;

use num_traits::Num;

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

#[derive(PartialEq, Eq, Debug)]
pub enum IIRFilterType {
    Normal,
    SecondOrder,
}

pub struct IIRFilter<C, T> {
    iirtype: IIRFilterType,
    buffer: Window<T>,
    numerator_coefs: DotProduct<C>,
    denominator_coefs: DotProduct<C>,
    second_order_sections: Vec<SecondOrderFilter<C, T>>,
}

impl<C: Copy + Num + Sum, T: Copy> IIRFilter<C, T> {
    /// Creates and IIR Filter with coeffients of type `C` and takes in data of type `T`
    ///
    /// It should be noted that `T` and `C` are both numerican and can be multiplied and added together.
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir_filter::*;
    /// use num::complex::Complex;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::<f64, Complex<f64>>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    /// ```
    pub fn new(
        feed_forward: &[C],
        feed_back: &[C],
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

    /// Executes type `T` and returns the data type `Out`
    ///
    /// `Out` is whatever the data type results in the multiplication of `C` and `T`
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir_filter::*;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let output = iir_filter.execute(1f64);
    ///
    /// assert_eq!(output, 0.05816769596076701);
    ///
    /// ```
    pub fn execute<Out>(&mut self, input: T) -> Out
    where
        DotProduct<C>: Execute<T, Output = Out>,
        T: Sub<Out, Output = T>,
        Out: Sub<Out, Output = T>,
    {
        match self.iirtype {
            IIRFilterType::Normal => {
                let buffer = self.buffer.to_vec();
                let denom_output =
                    Execute::execute(&self.denominator_coefs, &buffer[..(buffer.len() - 1)]);
                let mixed_output = input - denom_output;

                self.buffer.push(mixed_output);

                Execute::execute(&self.numerator_coefs, &self.buffer.to_vec())
            }
            IIRFilterType::SecondOrder => {
                let mut int_output = self.second_order_sections[0].execute::<Out>(Left(input));
                for second_order_filter in self.second_order_sections[1..].iter_mut() {
                    int_output = second_order_filter.execute(Right(int_output));
                }
                int_output
            }
        }
    }

    /// Executes array of type `T` and returns an array of the data type `Out`
    ///
    /// `Out` is whatever the data type results in the multiplication of `C` and `T`
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir_filter::*;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let output = iir_filter.execute_block(&[1.0, 0.0, 1.0, 0.0, 1.0]);
    ///
    /// assert_eq!(output, [0.05816769596076701, 0.119535296293297, 0.18410279587774706, 0.2518701895942824, 0.32283747232307686]);
    ///
    /// ```
    pub fn execute_block<Out>(&mut self, samples: &[T]) -> Vec<Out>
    where
        DotProduct<C>: Execute<T, Output = Out>,
        T: Sub<Out, Output = T>,
        Out: Sub<Out, Output = T>,
    {
        let mut block: Vec<Out> = vec![];
        for &sample in samples.iter() {
            block.push(self.execute(sample));
        }
        block
    }
    /// Returns the Numerator Coefs that the second order filter is using
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir_filter::*;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::<f64, f64>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let numerators = iir_filter.numerator_coefs();
    ///
    /// assert_eq!(numerators.to_vec(),  filter.0);
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
    /// use solid::filter::iirdes;
    /// use solid::filter::iir_filter::*;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::<f64, f64>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let denominators = iir_filter.denominator_coefs();
    ///
    /// assert_eq!(denominators.to_vec(),  filter.1);
    /// ```
    #[inline(always)]
    pub fn denominator_coefs(&self) -> &Vec<C> {
        self.denominator_coefs.coefficents()
    }

    /// Returns all the Second Order Internal Filters
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir_filter::*;
    ///
    /// let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0).unwrap();
    /// let mut iir_filter = IIRFilter::<f64, f64>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder).unwrap();
    ///
    /// let filters = iir_filter.second_order_filters();
    ///
    /// assert_eq!(filters.len(), 1);
    /// ```
    #[inline(always)]
    pub fn second_order_filters(&self) -> &Vec<SecondOrderFilter<C, T>> {
        &self.second_order_sections
    }

    /// Returns the IIR Type, Most times this should just be a second order filter
    ///
    /// Example
    ///
    /// ```
    /// use solid::filter::iirdes;
    /// use solid::filter::iir_filter::*;
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

impl<C: fmt::Display, T: fmt::Display> fmt::Display for IIRFilter<C, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IIR<{}>", std::any::type_name::<C>())
    }
}