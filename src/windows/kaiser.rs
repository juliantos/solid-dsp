//! Kaiser Window
//! 
//! James F. Kaiser and Ronald W. Schafer, "On
//! the Use of I0-Sinh Window for Spectrum Analysis,"
//! IEEE Transactions on Acoustics, Speech, and Signal
//! Processing, vol. ASSP-28, no. 1, pp. 105--107,
//! February, 1980.
use super::{WindowError, WindowErrorCode};
use super::super::math::Bessel;

use std::error::Error;

/// Function to calculate the value of the tap at `index` of `window_length`
/// 
/// Uses the Kaiser formula to generate taps in the window
/// 
/// # Example
/// 
/// ```
/// use solid::windows::kaiser;
/// 
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// for i in 0..window_len {
///     h[i] = match kaiser::kaiser(i, window_len, 8.6) {
///         Ok(val) => val,
///         _ => 0.0
///     };
///     assert_ne!(h[i], 0.0);
/// }
/// 
/// ```
pub fn kaiser(index: usize, window_length: usize, beta: f64) -> Result<f64, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)));
    } else if beta < 0.0 {
        return Err(Box::new(WindowError(WindowErrorCode::BetaLessThanZero)))
    }

    let t = index as f64 - (window_length - 1) as f64 / 2.0;
    let r = 2.0 * t / ((window_length - 1) as f64);
    let a = (beta * (1.0 - r * r).sqrt()).besseli(0.0);
    let b = beta.besseli(0.0);

    Ok(a / b)
}