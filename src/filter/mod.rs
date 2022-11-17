pub mod auto_correlator;
pub mod fir;
pub mod firdes;
pub mod iir;
pub mod iirdes;

use num::Complex;

pub trait Filter<I, O> {
    /// Executes type `T` and returns the data type `O`
    ///
    /// `O` is whatever the data type results in the multiplication of `C` and `T`
    fn execute(&mut self, sample: I) -> Vec<O>;
    /// Executes array of type `T` and returns an array of the data type `O`
    ///
    /// `O` is whatever the data type results in the multiplication of `C` and `T`
    fn execute_block(&mut self, samples: &[I]) -> Vec<O>;
    /// Computes the Complex Frequency response of the filter
    fn frequency_response(&self, frequency: f64) -> Complex<f64>;
    /// Computes the Group Delay in samples
    fn group_delay(&self, frequency: f64) -> f64;
}