use super::super::super::dot_product::{DotProduct, Direction, execute::Execute};
use super::super::super::window::Window;

use std::fmt;
use std::error::Error;
use std::iter::Sum;

use num_traits::Num;

#[derive(Debug)]
pub enum SecondOrderErrorCode {
    CoefficientsNotInRange
}

#[derive(Debug)]
pub struct SecondOrderError(pub SecondOrderErrorCode);

impl fmt::Display for SecondOrderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Second Order Error {:?}", self.0)
    }
}

impl Error for SecondOrderError {}

pub struct SecondOrder<C, In, Out> {
    input_buffer_i: Vec<In>,
    output_buffer_i: Vec<Out>,
    form_buffer_ii: Window<Out>,
    numerator_coefs: DotProduct<C>,
    denominator_coefs: DotProduct<C>
}

impl<C: Copy + Num + Sum, In: Copy, Out: Copy> SecondOrder<C, In, Out> {
    pub fn new(feed_forward: &[C], feed_back: &[C]) -> Result<Self, Box<dyn Error>> {
        if feed_forward.len() < 3 {
            return Err(Box::new(SecondOrderError(SecondOrderErrorCode::CoefficientsNotInRange)));
        } else if feed_back.len() < 3 {
            return Err(Box::new(SecondOrderError(SecondOrderErrorCode::CoefficientsNotInRange)))
        }

        let a0 = feed_back[0];
        let b = [feed_forward[0] / a0, feed_forward[1] / a0, feed_forward[2] / a0];
        let a = [feed_back[0] / a0, feed_back[1] / a0, feed_back[2] / a0];
        Ok(SecondOrder {
            input_buffer_i: Vec::new(),
            output_buffer_i: Vec::new(),
            form_buffer_ii: Window::new(3, 0),
            numerator_coefs: DotProduct::new(&a, Direction::FORWARD),
            denominator_coefs: DotProduct::new(&b, Direction::FORWARD)
        })
    }

    pub fn execute_df2(&mut self, input: In) -> Out {
        
    }
}