//! An Infinite Inpulse Response Filter
//! 
//! # Example
//! 
//! ```
//! ```

pub mod second_order;

use second_order::SecondOrder;
use super::super::dot_product::{DotProduct, Direction, execute::Execute};
use super::super::window::Window;

use std::fmt;
use std::error::Error;
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
    SecondOrderSectionSizeNotMultpleOf3
}

#[derive(Debug)]
pub struct IIRError(pub IIRErrorCode);

impl fmt::Display for IIRError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IIR Filter Error {:?}", self.0)
    }
}

impl Error for IIRError {}

pub enum IIRFilterType {
    Normal,
    SecondOrder
}

pub struct IIRFilter<C, T> {
    iirtype: IIRFilterType,
    buffer: Window<T>,
    numerator_coefs: DotProduct<C>,
    denominator_coefs: DotProduct<C>,
    second_order_sections: Vec<SecondOrder<C, T>>,
}

impl<C: Copy + Num + Sum, T: Copy> IIRFilter<C, T> {
    pub fn new(feed_forward: &[C], feed_back: &[C], iirtype: IIRFilterType) -> Result<Self, Box<dyn Error>> {
        match iirtype {
            IIRFilterType::Normal => {
                if feed_forward.len() == 0 {
                    return Err(Box::new(IIRError(IIRErrorCode::NumeratorLengthZero)))
                } else if feed_back.len() == 0 {
                    return Err(Box::new(IIRError(IIRErrorCode::DenominatorLengthZero)))
                }
        
                let window_length;
                if feed_back.len() > feed_forward.len() {
                    window_length = feed_back.len();
                } else {
                    window_length = feed_forward.len();
                }
        
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
        
                return Ok(IIRFilter {
                    iirtype: iirtype, 
                    buffer: buffer,
                    numerator_coefs: DotProduct::new(&numerator, Direction::FORWARD),
                    denominator_coefs: DotProduct::new(&denominator[1..], Direction::FORWARD),
                    second_order_sections: Vec::new()
                })
            }
            IIRFilterType::SecondOrder => {
                if feed_forward.len() != feed_back.len() {
                    return Err(Box::new(IIRError(IIRErrorCode::SecondOrderSectionSizeMismatch)))
                } else if feed_forward.len() == 0{
                    return Err(Box::new(IIRError(IIRErrorCode::SecondOrderSectionSizeZero)))
                } else if feed_forward.len() % 3 != 0 {
                    return Err(Box::new(IIRError(IIRErrorCode::SecondOrderSectionSizeNotMultpleOf3)))
                }

                let len = feed_forward.len() / 3;
                let buffer = Window::new(len * 2, 0);

                let mut second_order_vector = Vec::new();
                for i in 0..len {
                    second_order_vector.push(SecondOrder::new(&feed_forward[(3*i)..(3*i+3)], &feed_back[(3*i)..(3*i+3)])?);
                }

                Ok(IIRFilter {
                    iirtype: iirtype,
                    buffer: buffer,
                    numerator_coefs: DotProduct::new(&feed_forward, Direction::FORWARD),
                    denominator_coefs: DotProduct::new(&feed_back, Direction::FORWARD),
                    second_order_sections: second_order_vector
                })
            }
        }

    }

    pub fn execute<Out>(&mut self, input: T) -> Out 
    where DotProduct<C>: Execute<T, Output=Out>,
          T: Sub<Out, Output=T>,
          Out: Sub<Out, Output=T>
    {
        match self.iirtype {
            IIRFilterType::Normal => {
                let buffer = self.buffer.to_vec();
                let denom_output = Execute::execute(&self.denominator_coefs, &buffer[..(buffer.len() - 1)]);
                let mixed_output = input - denom_output;
        
                self.buffer.push(mixed_output);
        
                let numer_output = Execute::execute(&self.numerator_coefs, &self.buffer.to_vec());
                numer_output
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
}