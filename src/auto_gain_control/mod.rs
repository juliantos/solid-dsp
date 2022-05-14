//! An Automatic Gain Controller Component to Normalize the level of the incoming signal written [`AGC`]
//! 
//! Normalizing the incoming signal is a vital component to Digital Signal Processing before any further
//! steps can be taken. It implements a transfer function that is linear at a low signal level and a high
//! signal level. When the signal is inside the transfer function's threshold the AGC becomes active and 
//! target signal is held at unity. 
//! 
//! # Locking
//! 
//! The AGC object is able to be locked to the current gain. This is useful in short burst applications
//! when after receiving a header. With QAM Applications this reduces the symbol error.
//! 
//! # Squelch
//! 
//! The AGC Object has the ability to squelch any operation on the incoming signal if the threshold is not
//! met. This prevents random noise from the signal floor from being randomly amplified. 
//! 
//! # Example
//! 
//! ```
//! use solid::auto_gain_control::AGC;
//! use num::Complex;
//! 
//! let len = 500;
//! let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.05).collect();
//! let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.05).collect();
//! let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();
//! 
//! // The RSSI of this signal is about -26 so setting a threshold of -30 should enable all the way
//! let mut agc = AGC::new();
//! agc.squelch_enable();
//! agc.squelch_set_threshold(-30.0);
//! agc.set_bandwidth(0.02).unwrap();
//! let agc_vec = agc.execute_block(&complex_vec);
//! 
//! let last_item = agc_vec[agc_vec.len() - 1];
//! let val = (last_item.re.powf(2.0) + last_item.im.powf(2.0)).sqrt();
//! 
//! assert!(val > 0.98 && val < 1.02);
//! assert!(agc.get_rssi() < -25.5 && agc.get_rssi() > -26.0);
//! ```

use super::math::complex::{Conj, Real};

use std::fmt;
use std::ops::{Mul};
use std::error::Error;

#[derive(Debug, PartialEq)]
pub enum AGCErrorCode {
    BandwidthOutOfRange,
    SignalLevelOutOfRange,
    GainBelowThreshold,
    ScaleBelowThreshold,
    SamplesTooLow
}

#[derive(Debug)]
pub struct AGCError(pub AGCErrorCode, f64);

impl fmt::Display for AGCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let error_code = match self.0 {
            AGCErrorCode::BandwidthOutOfRange => self.1.to_string() + " Bandwidth not in range [0, 1]",
            AGCErrorCode::SignalLevelOutOfRange => self.1.to_string() + " Level is too low (0, inf)",
            AGCErrorCode::GainBelowThreshold => self.1.to_string() + " Gain is below Threshold (0, inf)",
            AGCErrorCode::ScaleBelowThreshold => self.1.to_string() + " Scale is below Threshold (0, inf)",
            AGCErrorCode::SamplesTooLow => "Need more than 0 Samples to operate".to_string()
        };
        write!(f, "AGC Error {}", error_code)
    }
}

impl Error for AGCError {}

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum SquelchMode {
    UNKNOWN,
    ENABLED,
    RISE,
    SIGNALHI,
    FALL,
    SINGALLO,
    TIMEOUT,
    DISABLED
}

#[allow(dead_code)]
pub struct AGC {
    gain: f64,
    scale: f64,
    bandwidth: f64,
    alpha: f64,
    energy_estimate: f64,
    lock: bool,
    squelch_mode: SquelchMode,
    squelch_threshold: f64,
    squelch_timeout: usize,
    squelch_timer: usize
}

impl AGC {
    /// Constructs a new `AGC` with the default gain of 1.0, scale of 1.0, bandwith of 0.1
    /// and squelch disabled.
    /// 
    /// The `AGC` is initially unlocked.
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// use num::Complex;
    /// 
    /// let len = 500;
    /// let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.05).collect();
    /// let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.05).collect();
    /// let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();
    /// 
    /// // The RSSI of this signal is about -27 so setting a threshold of -30 should enable all the way
    /// let mut agc = AGC::new();
    /// agc.squelch_enable();
    /// agc.squelch_set_threshold(-30.0);
    /// agc.set_bandwidth(0.01).unwrap();
    /// let agc_vec = agc.execute_block(&complex_vec);
    /// 
    /// assert!(agc.get_signal_level() < 0.05);
    /// ```
    pub fn new() -> Self {
        AGC {
            gain: 1.0,
            scale: 1.0,
            bandwidth: 0.1,
            alpha: 0.1,
            energy_estimate: 1.0,
            lock: false,
            squelch_mode: SquelchMode::DISABLED,
            squelch_threshold: 0.0,
            squelch_timeout: 100,
            squelch_timer: 0
        }
    }

    /// Resets the `AGC` to its initial state
    /// 
    /// Keeps the Squelch enabled if it was on 
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// use num::Complex;
    /// 
    /// let len = 500;
    /// let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.05).collect();
    /// let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.05).collect();
    /// let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();
    /// 
    /// // The RSSI of this signal is about -27 so setting a threshold of -30 should enable all the way
    /// let mut agc = AGC::new();
    /// agc.squelch_enable();
    /// agc.squelch_set_threshold(-30.0);
    /// agc.set_bandwidth(0.01).unwrap();
    /// let agc_vec = agc.execute_block(&complex_vec);
    /// 
    /// assert!(agc.get_gain() > 1.0);
    /// agc.reset();
    /// assert!(agc.get_gain() == 1.0);
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.gain = 1.0;
        self.energy_estimate = 1.0;
        self.lock = false;

        if self.squelch_mode == SquelchMode::DISABLED {
            self.squelch_mode = SquelchMode::DISABLED;
        } else {
            self.squelch_mode = SquelchMode::ENABLED;
        }
    }

    /// Executes Automatic Gain Control on a single sample
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// use num::Complex;
    /// 
    /// let len = 500;
    /// let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.05).collect();
    /// let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.05).collect();
    /// let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();
    /// 
    /// // The RSSI of this signal is about -27 so setting a threshold of -30 should enable all the way
    /// let mut agc = AGC::new();
    /// agc.squelch_enable();
    /// agc.squelch_set_threshold(-30.0);
    /// agc.set_bandwidth(0.01).unwrap();
    /// let agc_samp_out = agc.execute(complex_vec[0]);
    /// let second_samp = agc.execute(complex_vec[1]);
    /// 
    /// assert_eq!(agc_samp_out, complex_vec[0]);
    /// assert!(second_samp != complex_vec[1]);
    /// ```
    pub fn execute<T: Mul<f64, Output=T> + Mul<T, Output=T> + Conj + Real<Output=f64> + Copy>(&mut self, input: T) -> T {
        // First get the output
        let out: T = input * self.gain;

        // Calculate the new energy and update the estimate
        let ee: f64 = (out.conj() * out).real();
        self.energy_estimate = (1.0 - self.alpha) * self.energy_estimate + ee * self.alpha;

        // If locked then return early
        if self.lock { return out }

        if self.energy_estimate > 0.000001 {
            self.gain *= (-0.5 * self.alpha * (self.energy_estimate).ln()).exp();
        }

        if self.gain > 1000000.0 {
            self.gain = 1000000.0;
        }

        self.update_squelch_mode();

        if self.squelch_mode == SquelchMode::ENABLED { return input }
        out * self.scale
    }

    /// Executes the Automatic Gain Control on a slice of Samples
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// use num::Complex;
    /// 
    /// let len = 500;
    /// let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.05).collect();
    /// let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.05).collect();
    /// let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();
    /// 
    /// // The RSSI of this signal is about -27 so setting a threshold of -30 should enable all the way
    /// let mut agc = AGC::new();
    /// agc.squelch_enable();
    /// agc.squelch_set_threshold(-30.0);
    /// agc.set_bandwidth(0.01).unwrap();
    /// let agc_vec = agc.execute_block(&complex_vec);
    /// 
    /// assert_eq!(agc_vec.len(), complex_vec.len());
    /// assert!(agc_vec != complex_vec);
    /// assert_eq!(agc_vec[0], complex_vec[0]);
    /// ```
    #[inline]
    pub fn execute_block<T: Mul<f64, Output=T> + Mul<T, Output=T> + Conj + Real<Output=f64> + Copy>(&mut self, input: &[T]) -> Vec<T> {
        let mut out: Vec<T> = vec!();
        for &i in input.iter() {
            out.push(self.execute(i));
        }

        out
    }


    /// Locks the gain
    /// 
    /// # Example 
    ///
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let mut agc = AGC::new();
    /// 
    /// assert_eq!(agc.is_unlocked(), false);
    /// agc.lock();
    /// assert_eq!(agc.is_unlocked(), true);
    /// agc.unlock();
    /// assert_eq!(agc.is_unlocked(), false);
    /// ```
    #[inline]
    pub fn lock(&mut self) { 
        self.lock = true 
    }

    /// Unlocks the gain
    /// 
    /// # Example 
    ///
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let mut agc = AGC::new();
    /// 
    /// assert_eq!(agc.is_unlocked(), false);
    /// agc.lock();
    /// assert_eq!(agc.is_unlocked(), true);
    /// agc.unlock();
    /// assert_eq!(agc.is_unlocked(), false);
    #[inline]
    pub fn unlock(&mut self) { 
        self.lock = false 
    }

    /// Gets if the gain is unlocked
    /// 
    /// # Example 
    ///
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let mut agc = AGC::new();
    /// 
    /// assert_eq!(agc.is_unlocked(), false);
    /// agc.lock();
    /// assert_eq!(agc.is_unlocked(), true);
    /// agc.unlock();
    /// assert_eq!(agc.is_unlocked(), false);
    #[inline]
    pub fn is_unlocked(&self) -> bool {
        self.lock
    }

    /// Gets the loop bandwidth of the `AGC`
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let agc = AGC::new();
    /// 
    /// assert_eq!(agc.get_bandwidth(), 0.1);
    /// ```
    #[inline]
    pub fn get_bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Sets the loop bandwith of the `AGC`
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let mut agc = AGC::new();
    /// agc.set_bandwidth(0.01).unwrap();
    /// 
    /// assert_eq!(agc.get_bandwidth(), 0.01);
    /// ```
    #[inline]
    pub fn set_bandwidth(&mut self, bandwidth: f64) -> Result<f64, Box<dyn Error>> {
        if bandwidth < 0.0 || bandwidth > 1.0 {
            return Err(Box::new(AGCError(AGCErrorCode::BandwidthOutOfRange, bandwidth)))
        }

        self.bandwidth = bandwidth;
        self.alpha = bandwidth;

        Ok(bandwidth)
    }

    /// Gets the linear signal level 
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let agc = AGC::new();
    /// 
    /// assert_eq!(agc.get_signal_level(), 1.0);
    /// ```
    #[inline]
    pub fn get_signal_level(&self) -> f64 {
        1.0 / self.gain
    }

    /// Sets the linear signal level 
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let mut agc = AGC::new();
    /// agc.set_signal_level(10.0);
    /// 
    /// assert_eq!(agc.get_signal_level(), 10.0);
    /// ```
    pub fn set_signal_level(&mut self, level: f64) -> Result<f64, Box<dyn Error>> {
        if level <= 0.0 {
            return Err(Box::new(AGCError(AGCErrorCode::SignalLevelOutOfRange, level)))
        }

        self.gain = 1.0 / level;
        self.energy_estimate = 1.0;

        Ok(level)
    }

    /// Gets the estimated signal level in dB
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let mut agc = AGC::new();
    /// 
    /// assert_eq!(agc.get_rssi(), -0.0);
    /// ```
    #[inline]
    pub fn get_rssi(&self) -> f64 {
        self.gain.log10() * -20.0
    }

    /// Sets the estimated signal level in dB
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let mut agc = AGC::new();
    /// agc.set_rssi(-20.0);
    /// 
    /// assert_eq!(agc.get_rssi(), -20.0);
    /// ```
    pub fn set_rssi(&mut self, rssi: f64) {
        self.gain = 10f64.powf(- rssi / 20f64);
        
        if self.gain < 10f64.powi(-16) {
            self.gain = 10f64.powi(-16)
        }

        self.energy_estimate = 1.0;
    }

    /// Gets the internal gain of the `AGC`
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let agc = AGC::new();
    /// 
    /// assert_eq!(agc.get_gain(), 1.0);
    /// ```
    #[inline]
    pub fn get_gain(&self) -> f64 {
        self.gain
    }

    /// Sets the internal gain of the `AGC`
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let mut agc = AGC::new();
    /// agc.set_gain(2.0);
    /// 
    /// assert_eq!(agc.get_gain(), 2.0);
    /// ```
    #[inline]
    pub fn set_gain(&mut self, gain: f64) -> Result<f64, Box<dyn Error>> {
        if gain <= 0.0 {
            return Err(Box::new(AGCError(AGCErrorCode::GainBelowThreshold, gain)))
        }

        self.gain = gain;
        Ok(gain)
    }

    /// Gets the internal scale of the `AGC`
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let agc = AGC::new();
    /// 
    /// assert_eq!(agc.get_scale(), 1.0);
    /// ```
    #[inline]
    pub fn get_scale(&self) -> f64 {
        self.scale
    }

    /// Sets the internal scale of the `AGC`
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// 
    /// let mut agc = AGC::new();
    /// agc.set_scale(2.0);
    /// 
    /// assert_eq!(agc.get_scale(), 2.0);
    /// ```
    #[inline]
    pub fn set_scale(&mut self, scale: f64) -> Result<f64, Box<dyn Error>> {
        if scale <= 0.0 {
            return Err(Box::new(AGCError(AGCErrorCode::ScaleBelowThreshold, scale)))
        }

        self.scale = scale;
        Ok(scale)
    }

    /// Initializes the internal gain of the `AGC` using some samples
    /// 
    /// Returns the resulting linear signal level
    /// 
    /// # Example
    /// 
    /// ```
    /// use solid::auto_gain_control::AGC;
    /// use num::Complex;
    /// 
    /// let len = 500;
    /// let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.05).collect();
    /// let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.05).collect();
    /// let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();
    /// 
    /// // The RSSI of this signal is about -27 so setting a threshold of -30 should enable all the way
    /// let mut agc = AGC::new();
    /// agc.squelch_enable();
    /// agc.squelch_set_threshold(-30.0);
    /// agc.set_bandwidth(0.01).unwrap();
    /// let signal_level = agc.init(&complex_vec).unwrap();
    /// 
    /// assert!(signal_level > 0.04999 && signal_level <= 0.05);
    /// ```
    pub fn init<T: Mul<f64, Output=T> + Mul<T, Output=T> + Conj + Real<Output=f64> + Copy>(&mut self, input: &[T]) -> Result<f64, Box<dyn Error>> {
        if input.len() <= 0 {
            return Err(Box::new(AGCError(AGCErrorCode::SamplesTooLow, 0.0)))
        }

        let mut x2: f64 = 0.0;
        for &i in input.iter() {
            x2 += (i * i.conj()).real()
        }

        x2 = (x2 / input.len() as f64).sqrt() + 10f64.powi(-16);

        self.set_signal_level(x2)
    }

    #[inline]
    pub fn squelch_enable(&mut self) {
        self.squelch_mode = SquelchMode::ENABLED;
    }

    #[inline]
    pub fn squelch_disable(&mut self) {
        self.squelch_mode = SquelchMode::DISABLED;
    }

    pub fn is_squelch_enabled(&self) -> bool {
        if self.squelch_mode == SquelchMode::DISABLED {
            return false
        }

        true
    }

    #[inline]
    pub fn squelch_get_threshold(&self) -> f64 {
        self.squelch_threshold
    }

    #[inline]
    pub fn squelch_set_threshold(&mut self, threshold: f64) {
        self.squelch_threshold = threshold
    }

    #[inline]
    pub fn squelch_get_timeout(&self) -> usize {
        self.squelch_timeout
    }

    #[inline]
    pub fn squelch_set_timeout(&mut self, timeout: usize) {
        self.squelch_timeout = timeout
    }

    #[inline]
    pub fn squelch_get_mode(&self) -> SquelchMode {
        self.squelch_mode
    }

    pub fn update_squelch_mode(&mut self) {
        let threshold_exceeded;
        if self.get_rssi() > self.squelch_threshold {
            threshold_exceeded = true
        } else {
            threshold_exceeded = false
        }

        self.squelch_mode = match self.squelch_mode {
            SquelchMode::ENABLED => if threshold_exceeded { SquelchMode::RISE } else { SquelchMode::ENABLED },
            SquelchMode::RISE => if threshold_exceeded { SquelchMode::SIGNALHI } else { SquelchMode::FALL },
            SquelchMode::SIGNALHI => if threshold_exceeded { SquelchMode::SIGNALHI } else { SquelchMode::FALL },
            SquelchMode::FALL => {self.squelch_timer = self.squelch_timeout; if threshold_exceeded { SquelchMode::SIGNALHI } else { SquelchMode::SINGALLO }},
            SquelchMode::SINGALLO => {self.squelch_timer -= 1; if self.squelch_timer == 0 { SquelchMode::TIMEOUT } 
                else if threshold_exceeded { SquelchMode::SIGNALHI } else { SquelchMode::SINGALLO} },
            SquelchMode::TIMEOUT => SquelchMode::ENABLED,
            _ => SquelchMode::DISABLED
        };
    }
}