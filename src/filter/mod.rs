pub mod auto_correlator;
pub mod fir;
pub mod firdes;
pub mod iir;
pub mod iirdes;

use num::Complex;

pub trait Filter<C, T, Out> {
    /// Executes type `T` and returns the data type `Out`
    ///
    /// `Out` is whatever the data type results in the multiplication of `C` and `T`
    fn execute(&mut self, sample: T) -> Vec<Out>;
    /// Executes array of type `T` and returns an array of the data type `Out`
    ///
    /// `Out` is whatever the data type results in the multiplication of `C` and `T`
    fn execute_block(&mut self, samples: &[T]) -> Vec<Out>;
    /// Computes the Complex Frequency response of the filter
    fn frequency_response(&self, frequency: f64) -> Complex<f64>;
    /// Computes the Group Delay in samples
    fn group_delay(&self, frequency: f64) -> f64;
}