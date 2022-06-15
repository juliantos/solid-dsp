//! Hann window 
use super::{WindowError, WindowErrorCode};

use std::error::Error;

/// Function to calculate the value of the tap at `index` of `window_length`
/// 
/// Uses the Hann formula to generate taps in the window
/// 
/// # Example
/// 
/// ```
/// use solid::windows::hann;
/// 
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// for i in 0..window_len {
///     h[i] = match hann::hann(i, window_len) {
///         Ok(val) => val,
///         _ => -10.0
///     };
///     assert_ne!(h[i], -10.0);
/// }
/// 
/// ```
pub fn hann(index: usize, window_length: usize) -> Result<f64, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)));
    }

    Ok(0.5 - 0.5 * ((2.0 * std::f64::consts::PI * index as f64) / ((window_length - 1) as f64)).cos())
}