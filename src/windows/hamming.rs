//! Hamming window 
//! 
//! Albert H. Nuttall, "Some Windows with Very Good Sidelobe
//! Behavior,"  IEEE Transactions on Acoustics, Speech, and Signal
//! Processing, vol. ASSP-29, no. 1, pp. 84-91, February, 1981.
use super::{WindowError, WindowErrorCode};

use std::error::Error;

/// Function to calculate the value of the tap at `index` of `window_length`
/// 
/// Uses the Hamming formula to generate taps in the window
/// 
/// # Example
/// 
/// ```
/// use solid::windows::hamming;
/// 
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// for i in 0..window_len {
///     h[i] = match hamming::hamming(i, window_len) {
///         Ok(val) => val,
///         _ => 0.0
///     };
///     assert_ne!(h[i], 0.0);
/// }
/// 
/// ```
pub fn hamming(index: usize, window_length: usize) -> Result<f64, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)));
    }

    Ok(0.53836 - 0.46164 * ((2.0 * std::f64::consts::PI * index as f64) / ((window_length - 1) as f64)).cos())
}