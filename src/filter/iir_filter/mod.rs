//! An Infinite Inpulse Response Filter
//! 
//! # Example
//! 
//! ```
//! ```

pub mod second_order;

use second_order::SecondOrder;
use super::super::dot_product::{DotProduct, Direction, execute::Execute};

use std::fmt;
use std::iter::Sum;

use num_traits::Num;

#[allow(dead_code)]
pub struct IIRFilter<C, In, Out> {
    buffer: Vec<In>, // Might be Window
    numerator_dot_prod: DotProduct<C>,
    denominator_dot_prod: DotProduct<C>,
    second_order_sections: Vec<SecondOrder<C, In, Out>>,
}

// impl<C: Copy + Num + Sum, In: Copy, Out> IIRFilter<C, In, Out> {
//     pub fn new(feed_forward: &[C], feed_back: &[C]) -> Self {
//         IIRFilter {
//             buffer: Vec::new(),
//         }
//     }
// }