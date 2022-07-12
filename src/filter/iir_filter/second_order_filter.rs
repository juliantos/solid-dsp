// TODO: Documentation, Group Delay. Print Statements

use super::super::super::dot_product::{DotProduct, Direction, execute::Execute};
use super::super::super::window::Window;

use std::fmt;
use std::error::Error;
use std::iter::Sum;
use std::ops::Sub;

use either::Either;

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

pub struct SecondOrderFilter<C, T> {
    form_buffer_ii: Window<T>,
    numerator_coefs: DotProduct<C>,
    denominator_coefs: DotProduct<C>
}

impl<C: Copy + Num + Sum, T: Copy> SecondOrderFilter<C, T> {
    pub fn new(feed_forward: &[C], feed_back: &[C]) -> Result<Self, Box<dyn Error>> {
        if feed_forward.len() < 3 {
            return Err(Box::new(SecondOrderError(SecondOrderErrorCode::CoefficientsNotInRange)));
        } else if feed_back.len() < 3 {
            return Err(Box::new(SecondOrderError(SecondOrderErrorCode::CoefficientsNotInRange)))
        }

        let a0 = feed_back[0];
        let b = [feed_forward[0] / a0, feed_forward[1] / a0, feed_forward[2] / a0];
        let a = [feed_back[0] / a0, feed_back[1] / a0, feed_back[2] / a0];

        Ok(SecondOrderFilter {
            form_buffer_ii: Window::new(3, 0),
            numerator_coefs: DotProduct::new(&a[1..], Direction::FORWARD),
            denominator_coefs: DotProduct::new(&b, Direction::FORWARD)
        })
    }

    pub fn execute<Out>(&mut self, input: Either<T, Out>) -> Out
    where DotProduct<C>: Execute<T, Output=Out> ,
          T: Sub<Out, Output=T>,
          Out: Sub<Out, Output=T>
    {
        let mut buffer = self.form_buffer_ii.to_vec();
        buffer[2] = buffer[1];
        buffer[1] = buffer[0];

        let denom_output = Execute::execute(&self.numerator_coefs, &buffer[1..]);

        let mixed_output;
        if input.is_left() {
            mixed_output = input.left().unwrap() - denom_output;
        } else {
            mixed_output = input.right().unwrap() - denom_output;
        }
        
        self.form_buffer_ii.push(mixed_output);
        let buffer = self.form_buffer_ii.to_vec();

        let numer_output = Execute::execute(&self.denominator_coefs, &buffer);
        numer_output
    }
    
    #[inline(always)]
    pub fn numerator_coefs(&self) -> &Vec<C> {
        self.numerator_coefs.coefficents()
    }

    #[inline(always)]
    pub fn denominator_coefs(&self) -> &Vec<C> {
        self.denominator_coefs.coefficents()
    }
}