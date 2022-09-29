//! Triangular window
use super::{WindowError, WindowErrorCode};

use std::error::Error;

/// Function to calculate the value of the tap at `index` of `window_length`
///
/// Uses the Triangular formula to generate taps in the window
///
/// # Example
///
/// ```
/// use solid::windows::triangular;
///
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// let n = 26;
/// for i in 0..window_len {
///     h[i] = match triangular::triangular(i, window_len, 26) {
///         Ok(val) => val,
///         _ => 0.0
///     };
///     assert_ne!(h[i], 0.0);
/// }
///
/// ```
pub fn triangular(index: usize, window_length: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)));
    } else if n != window_length - 1 && n != window_length && n != window_length + 1 {
        return Err(Box::new(WindowError(WindowErrorCode::TriangularSubLength)));
    } else if n == 0 {
        return Err(Box::new(WindowError(WindowErrorCode::TriangularZeroLength)));
    }

    let v0 = index as f64 - (window_length - 1) as f64 / 2.0;
    let v1 = n as f64 / 2.0;

    Ok(1.0 - (v0 / v1).abs())
}
