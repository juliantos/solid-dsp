//! Windowing functions used to calculate taps

pub mod hamming;
pub mod blackman_harris;
pub mod flattop;
pub mod kaiser;
//TODO: Hann, Triangular, RCosTaper, KBD

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum WindowErrorCode{
    OutOfBounds,
    BetaLessThanZero,
    EmptyWindow
}

#[derive(Debug)]
pub struct WindowError(WindowErrorCode);

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let error_code = match self.0 {
            WindowErrorCode::OutOfBounds => "Out of Bounds",
            WindowErrorCode::BetaLessThanZero => "Beta given is less than 0",
            WindowErrorCode::EmptyWindow => "Empty Window"
        };
        write!(f, "Window Error: {}", error_code)
    }
}

impl Error for WindowError {}

#[derive(Debug)]
pub enum Window {
    Unknown,
    Hamming,
    Blackmanharris,
    Blackmanharris7,
    Flattop,
    Kaiser
}