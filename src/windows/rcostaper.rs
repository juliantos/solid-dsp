//! RCosTaper window
use super::{WindowError, WindowErrorCode};

use std::error::Error;

/// Function to calculate the value of the tap at `index` of `window_length`
///
/// Uses the RCosTaper formula to generate taps in the window
///
/// # Example
///
/// ```
/// use solid::windows::rcostaper;
///
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// let taper = 12;
/// for i in 0..window_len {
///     h[i] = match rcostaper::rcostaper(i, window_len, taper) {
///         Ok(val) => val,
///         _ => 0.0
///     };
///     assert_ne!(h[i], 0.0);
/// }
///
/// ```
pub fn rcostaper(index: usize, window_length: usize, taper: usize) -> Result<f64, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)));
    } else if taper > window_length / 2 {
        return Err(Box::new(WindowError(WindowErrorCode::TaperLength)));
    }

    let temp_index = if index > window_length - taper - 1 {
        window_length - index - 1
    } else {
        index
    };

    if temp_index < taper {
        Ok(0.5 - 0.5 * ((std::f64::consts::PI * temp_index as f64 + 0.5) / (taper as f64)).cos())
    } else {
        Ok(1.0)
    }
}
