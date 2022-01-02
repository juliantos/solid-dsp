//! Finite Input Response Filter Design
//! 
//! 
pub mod filter_traits;

use super::super::math::{Sinc, Bessel};
use super::super::windows::kaiser;
use super::super::dot_product::{DotProduct, Direction, execute::Execute};

use std::error::Error;
use std::fmt;

use std::f64::consts::PI as PI_64;
use std::f32::consts::PI as PI_32;

use num::complex::Complex;

#[derive(Debug)]
enum FirdesErrorCode {
    InvalidBandwidth,
    InvalidStopBandLevel,
    InvalidMu,
    InvalidSemiLength,
    InvalidFilterSize,
    InvalidFFTSize
}

#[derive(Debug)]
struct FirdesError(FirdesErrorCode);

impl fmt::Display for FirdesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let error_code = match self.0 {
            FirdesErrorCode::InvalidBandwidth => "Invalid Bandwidth [0, 0.5]",
            FirdesErrorCode::InvalidStopBandLevel => "Invalid Stop Band Attenuation (0, inf)",
            FirdesErrorCode::InvalidMu => "Invalid Mu Range [-0.5, 0.5]",
            FirdesErrorCode::InvalidSemiLength => "Invalid Filter Semi Length [1, 1000]",
            FirdesErrorCode::InvalidFilterSize => "Invalid Filter Size [1, inf)",
            FirdesErrorCode::InvalidFFTSize => "Invalid FFT Size [1, inf)"
        };
        write!(f, "Firdes Error: {}", error_code)
    }
}

impl Error for FirdesError {}

pub enum EstimationMethod {
    Kaiser,
    Herrmann
}

/// Estimates the required filter length given transition bandwidth and stop-band attenuation
/// 
/// # Arguments
/// 
/// * `transitition_bandwidth` - value between 0.0 and 0.5 relative to the total bandwdith
/// * `stop_band_attenuation` - stop-band suppression level in dB
/// * `method` - The estimation method used (either [`EstimationMethod::Kaiser`] or [`EstimationMethod::Herrmann`])
/// 
/// # Example
/// 
/// ```
/// use solid::filter::firdes::{estimate_required_filter_length, EstimationMethod};
/// 
/// let est = match estimate_required_filter_length(0.35, 100.0, EstimationMethod::Herrmann) {
///     Ok(len) => len,
///     _ => 0
/// };
/// 
/// assert_eq!(est, 15);
/// ```
pub fn estimate_required_filter_length(transition_bandwidth: f64, stop_band_attenuation: f64, method: EstimationMethod) -> Result<usize, Box<dyn Error>> {
    if transition_bandwidth > 0.5 || transition_bandwidth < 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidBandwidth)))
    } else if stop_band_attenuation <= 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidStopBandLevel)))
    }

    let val = match method {
        EstimationMethod::Kaiser => estimate_required_filter_length_kaiser(transition_bandwidth, stop_band_attenuation),
        EstimationMethod::Herrmann => estimate_required_filter_length_herrmann(transition_bandwidth, stop_band_attenuation)
    };

    match val {
        Ok(val) => Ok(val as usize),
        Err(e) => Err(e)
    }
}

/// Estimates the required stop band attenuation given transition bandwidth and filter size
/// 
/// # Arguments
/// 
/// * `transition_bandwidth` - value between 0.0 and 0.5 relative to the total bandwidth
/// * `filter_length` - the length of the filter 
/// * `method` - The estimation method used (either [`EstimationMethod::Kaiser`] or [`EstimationMethod::Herrmann`])
/// 
/// # Example
/// 
/// ```
/// use solid::filter::firdes::{estimate_required_filter_stop_band_attenuation, EstimationMethod};
/// 
/// let est = match estimate_required_filter_stop_band_attenuation(0.35, 16, EstimationMethod::Herrmann) {
///     Ok(est) => est,
///     _ => 0.0
/// };
/// 
/// assert_eq!(est as usize, 101)
/// ```
pub fn estimate_required_filter_stop_band_attenuation(transition_bandwidth: f64, filter_length: usize, method: EstimationMethod) -> Result<f64, Box<dyn Error>> {
    // Based On Liquid DSP, search between these two stop-bands
    let mut as0 = 0.01;
    let mut as1 = 200.0;

    let mut as_hat = 0.0;
    for _ in 0..20 {
        as_hat = 0.5 * (as1 + as0);
        let n_hat = match method {
            EstimationMethod::Kaiser => estimate_required_filter_length_kaiser(transition_bandwidth, as_hat)?,
            EstimationMethod::Herrmann => estimate_required_filter_length_herrmann(transition_bandwidth, as_hat)?
        };

        if n_hat < filter_length as f64 {
            as0 = as_hat;
        } else {
            as1 = as_hat;
        }
    }

    Ok(as_hat)
}

/// Estimates the required filter transition bandwidth given stop band and filter length
/// 
/// # Arguments
/// 
/// * `stop_band_attenuation` - stop-band suppression level in dB
/// * `filter_length` - the length of the filter 
/// * `method` - The estimation method used (either [`EstimationMethod::Kaiser`] or [`EstimationMethod::Herrmann`])
/// 
/// # Example
/// 
/// ```
/// use solid::filter::firdes::{estimate_required_filter_transition, EstimationMethod};
/// 
/// let est = match estimate_required_filter_transition(101.0, 16, EstimationMethod::Herrmann) {
///     Ok(est) => (est + 0.005) * 100.0,
///     _ => 0.0
/// };
/// 
/// assert_eq!(est as usize, 35)
/// ```
pub fn estimate_required_filter_transition(stop_band_attenuation: f64, filter_length: usize, method: EstimationMethod) -> Result<f64, Box<dyn Error>> {
    let mut df0 = 0.001;
    let mut df1 = 0.499;

    let mut df_hat = 0.0;
    for _ in 0..20 {
        df_hat = 0.5 * (df1 + df0);
        let n_hat = match method {
            EstimationMethod::Kaiser => estimate_required_filter_length_kaiser(df_hat, stop_band_attenuation)?,
            EstimationMethod::Herrmann => estimate_required_filter_length_herrmann(df_hat, stop_band_attenuation)?
        };

        if n_hat < filter_length as f64 {
            df1 = df_hat;
        } else {
            df0 = df_hat;
        }
    }

    Ok(df_hat)
}

/// Estimates the filter length based on the kaiser method
pub fn estimate_required_filter_length_kaiser(transition_bandwidth: f64, stop_band_attenuation: f64) -> Result<f64, Box<dyn Error>> {
    if transition_bandwidth > 0.5 || transition_bandwidth < 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidBandwidth)))
    } else if stop_band_attenuation <= 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidStopBandLevel)))
    }

    Ok((stop_band_attenuation - 7.95) / (14.26 * transition_bandwidth))
}

/// Estimates the filter length based on the herrmann method
pub fn estimate_required_filter_length_herrmann(transition_bandwidth: f64, stop_band_attenuation: f64) -> Result<f64, Box<dyn Error>> {
    if transition_bandwidth > 0.5 || transition_bandwidth < 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidBandwidth)))
    } else if stop_band_attenuation <= 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidStopBandLevel)))
    }

    if stop_band_attenuation > 105.0 {
        return Ok((stop_band_attenuation - 7.95) / (14.26 * transition_bandwidth))
    }

    let new_stop_band_attenuation = stop_band_attenuation + 7.4;
    let d1 = 10f64.powf(-new_stop_band_attenuation / 20.0);
    let d2 = 10f64.powf(-new_stop_band_attenuation / 20.0);

    let t1 = d1.log10();
    let t2 = d2.log10();

    let d_inf = (0.005309*t1*t1 + 0.07114*t1 - 0.4761)*t2 - (0.002660*t1*t1 + 0.59410*t1 + 0.4278);

    let f = 11.012 + 0.51244*(t1-t2);

    Ok((d_inf - f*transition_bandwidth*transition_bandwidth) / transition_bandwidth + 1.0)    
}

/// Computes the Kaiser Window Beta factor given the target's stop-band attenuation
pub fn kaiser_beta(stop_band_attenuation: f64) -> f64 {
    let abs_as = stop_band_attenuation.abs();

    if abs_as > 50.0 {
        0.1102 * (abs_as - 8.7)
    } else if abs_as > 21.0 {
        0.5842 * (abs_as - 21.0).powf(0.4) + 0.07886 * (abs_as - 21.0)
    } else {
        0.0
    }
}

/// Design FIR Filter Coefficients using Kaiser Window
/// 
/// Creates a [`Vec<f64>`] of coefficients using the specified inputs.
/// 
/// # Arguments
/// 
/// * `filter_length` - size of the filter in taps
/// * `cutoff_frequency` - bandwidth of the frequency to remain from (0, 0.5)
/// * `stop_band_attenuation` - supression in dB (0, inf)
/// * `fraction_sample_offset` - fractional sample offset (-0.5, 0.5)
/// 
/// # Example 
/// 
/// ```
/// use solid::filter::firdes;
/// 
/// let taps = match firdes::firdes_kaiser(8, 0.35, 120.0, 0.0) {
///     Ok(taps) => taps,
///     _ => vec!()
/// };
/// 
/// assert_eq!(taps.len(), 8);
/// ```
pub fn firdes_kaiser(filter_length: usize, cutoff_frequency: f64, stop_band_attenuation: f64, fractional_sample_offset: f64) -> Result<Vec<f64>, Box<dyn Error>> {
    if fractional_sample_offset < -0.5 || fractional_sample_offset > 0.5 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidMu)))
    } else if cutoff_frequency > 0.5 || cutoff_frequency < 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidBandwidth)))
    } else if stop_band_attenuation <= 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidStopBandLevel)))
    }

    let beta = kaiser_beta(stop_band_attenuation);
    let mut h: Vec<f64> = vec![0.0; filter_length];
    for i in 0..filter_length {
        let t = i as f64 - ((filter_length - 1) as f64) / 2.0 + fractional_sample_offset;
        
        let h1 = (2.0 * cutoff_frequency * t).sinc();

        let h2 = kaiser::kaiser(i, filter_length, beta)?;

        h[i] = h1 * h2;
    }

    Ok(h)
}

/// Design FIR Filter Coefficients using Kaiser Window
///
/// Creates a [`Vec<f32>`] of coefficients using the specified inputs.
/// 
/// # Arguments
/// 
/// * `filter_length` - size of the filter in taps
/// * `cutoff_frequency` - bandwidth of the frequency to remain from (0, 0.5)
/// * `stop_band_attenuation` - supression in dB (0, inf)
/// * `fraction_sample_offset` - fractional sample offset (-0.5, 0.5)
/// 
/// # Example 
/// 
/// ```
/// use solid::filter::firdes;
/// 
/// let taps = match firdes::firdes_kaiser_f32(8, 0.35, 120.0, 0.0) {
///     Ok(taps) => taps,
///     _ => vec!()
/// };
/// 
/// assert_eq!(taps.len(), 8);
/// ```
pub fn firdes_kaiser_f32(filter_length: usize, cutoff_frequency: f32, stop_band_attenuation: f32, fractional_sample_offset: f32) -> Result<Vec<f32>, Box<dyn Error>> {
    if fractional_sample_offset < -0.5 || fractional_sample_offset > 0.5 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidMu)))
    } else if cutoff_frequency > 0.5 || cutoff_frequency < 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidBandwidth)))
    } else if stop_band_attenuation <= 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidStopBandLevel)))
    }

    let beta = kaiser_beta(stop_band_attenuation as f64) as f32;
    let mut h: Vec<f32> = vec![0.0; filter_length];
    for i in 0..filter_length {
        let t = i as f32 - ((filter_length - 1) as f32) / 2.0 + fractional_sample_offset;
        
        let h1 = (2.0 * cutoff_frequency * t).sinc();

        let h2 = kaiser::kaiser_f32(i, filter_length, beta)?;

        h[i] = h1 * h2;
    }

    Ok(h)
}

/// Design FIR Filter Coefficients based on the notch filter (Band-Stop)
///
/// Creates a [`Vec<f64>`] of coefficients using the specified inputs.
/// 
/// # Arguments
/// 
/// * `semi_length` - half size of the filter in taps
/// * `notch_frequency` - the filter normalized notch frequency (-0.5, 0.5)
/// * `stop_band_attenuation` - supression in dB (0, inf)
/// 
/// # Example 
/// 
/// ```
/// use solid::filter::firdes;
/// 
/// let taps = match firdes::firdes_notch(8, 0.35, 120.0) {
///     Ok(taps) => taps,
///     _ => vec!()
/// };
/// 
/// assert_eq!(taps.len(), 17);
/// ```
pub fn firdes_notch(semi_length: usize, notch_frequency: f64, stop_band_attenuation: f64) -> Result<Vec<f64>, Box<dyn Error>> {
    if semi_length < 1 || semi_length > 1000 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidSemiLength)))
    } else if notch_frequency > 0.5 || notch_frequency < 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidBandwidth)))
    } else if stop_band_attenuation <= 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidStopBandLevel)))
    }

    let beta = kaiser_beta(stop_band_attenuation);
    let h_len = 2 * semi_length + 1;

    let mut h = vec![0.0; h_len];
    let mut scale = 0.0;

    for i in 0..h_len {
        let tone = -(2.0 * PI_64 * notch_frequency * (i as f64 - semi_length as f64)).cos();
        let window = kaiser::kaiser(i, h_len, beta)?;
        h[i] = tone * window;
        scale += h[i] * tone;
    }

    // normalize
    for coef in h.iter_mut() {
        *coef /= scale;
    }

    // impulse
    h[semi_length] += 1.0;

    Ok(h)
}

/// Design FIR Filter Coefficients based on the notch filter (Band-Stop)
///
/// Creates a [`Vec<f32>`] of coefficients using the specified inputs.
/// 
/// # Arguments
/// 
/// * `semi_length` - half size of the filter in taps
/// * `notch_frequency` - the filter normalized notch frequency (-0.5, 0.5)
/// * `stop_band_attenuation` - supression in dB (0, inf)
/// 
/// # Example 
/// 
/// ```
/// use solid::filter::firdes;
/// 
/// let taps = match firdes::firdes_notch_f32(8, 0.35, 120.0) {
///     Ok(taps) => taps,
///     _ => vec!()
/// };
/// 
/// assert_eq!(taps.len(), 17);
/// ```
pub fn firdes_notch_f32(semi_length: usize, notch_frequency: f32, stop_band_attenuation: f32) -> Result<Vec<f32>, Box<dyn Error>> {
    if semi_length < 1 || semi_length > 1000 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidSemiLength)))
    } else if notch_frequency > 0.5 || notch_frequency < 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidBandwidth)))
    } else if stop_band_attenuation <= 0.0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidStopBandLevel)))
    }

    let beta = kaiser_beta(stop_band_attenuation as f64) as f32;
    let h_len = 2 * semi_length + 1;

    let mut h = vec![0.0; h_len];
    let mut scale = 0.0;

    for i in 0..h_len {
        let tone = -(2.0 * PI_32 * notch_frequency * (i as f32 - semi_length as f32)).cos();
        let window = kaiser::kaiser_f32(i, h_len, beta)?;
        h[i] = tone * window;
        scale += h[i] * tone;
    }

    // normalize
    for coef in h.iter_mut() {
        *coef /= scale;
    }

    // impulse
    h[semi_length] += 1.0;

    Ok(h)
}

/// Design FIR Filter Coefficients based on the doppler filter
///
/// Creates a [`Vec<f64>`] of coefficients using the specified inputs.
/// 
/// # Arguments
/// 
/// * `filter_length` - size of the filter in taps
/// * `doppler_frequency` - the filter normalized doppler frequency (0, 0.5)
/// * `rice_fading_factor` - `K` in dB (0, inf)
/// * `theta` - LoS component angle of arrival
/// 
/// # Example 
/// 
/// ```
/// use solid::filter::firdes;
/// 
/// let taps = match firdes::firdes_doppler(51, 0.1, 2.0, 0.0) {
///     Ok(taps) => taps,
///     _ => vec!()
/// };
/// 
/// assert_eq!(taps.len(), 51);
/// ```
pub fn firdes_doppler(filter_length: usize, doppler_frequency: f64, rice_fading_factor: f64, theta: f64) -> Result<Vec<f64>, Box<dyn Error>> {
    let beta = 4.0;

    let mut h = vec![0.0; filter_length];

    for i in 0..filter_length {
        // Time sample
        let t = i as f64 - (filter_length as f64 - 1.0) / 2.0;

        // Bessel
        let j = 1.5 * (2.0 * PI_64 * doppler_frequency * t).abs().besselj(0.0);

        // Rice-K component
        let r = 1.5 * rice_fading_factor / ( rice_fading_factor + 1.0) * (2.0 * PI_64 * doppler_frequency * t * theta.cos()).cos();

        // Window
        let w = kaiser::kaiser(i, filter_length, beta)?;

        // Composite
        h[i] = (j + r) * w;
    }

    Ok(h)
}

/// Design FIR Filter Coefficients based on the doppler filter
///
/// Creates a [`Vec<f32>`] of coefficients using the specified inputs.
/// 
/// # Arguments
/// 
/// * `filter_length` - size of the filter in taps
/// * `doppler_frequency` - the filter normalized doppler frequency (0, 0.5)
/// * `rice_fading_factor` - `K` in dB (0, inf)
/// * `theta` - LoS component angle of arrival
/// 
/// # Example 
/// 
/// ```
/// use solid::filter::firdes;
/// 
/// let taps = match firdes::firdes_doppler_f32(51, 0.1, 2.0, 0.0) {
///     Ok(taps) => taps,
///     _ => vec!()
/// };
/// 
/// assert_eq!(taps.len(), 51);
/// ```
pub fn firdes_doppler_f32(filter_length: usize, doppler_frequency: f32, rice_fading_factor: f32, theta: f32) -> Result<Vec<f32>, Box<dyn Error>> {
    let beta = 4.0;

    let mut h = vec![0.0; filter_length];

    for i in 0..filter_length {
        // Time sample
        let t = i as f32 - (filter_length as f32 - 1.0) / 2.0;

        // Bessel
        let j = 1.5 * (2.0 * PI_32 * doppler_frequency * t).abs().besselj(0.0);

        // Rice-K component
        let r = 1.5 * rice_fading_factor / ( rice_fading_factor + 1.0) * (2.0 * PI_32 * doppler_frequency * t * theta.cos()).cos();

        // Window
        let w = kaiser::kaiser_f32(i, filter_length, beta)?;

        // Composite
        h[i] = (j + r) * w;
    }

    Ok(h)
}

/// Computes the auto-correlation of a filter at a specific lag
/// 
/// # Arguments
/// 
/// * `filter` - the filter
/// * `lag` - auto-correlation lag (samples)
/// 
/// # Example
/// 
/// ```
/// use solid::filter::firdes::{filter_autocorrelation, firdes_notch};
/// 
/// let taps = match firdes_notch(25, 0.2, 30.0) {
///     Ok(taps) => taps,
///     _ => vec!()
/// };
/// 
/// let auto_corr = filter_autocorrelation(&taps, 3);
/// let rev_auto_corr = filter_autocorrelation(&taps, -3);
/// 
/// assert_eq!(auto_corr, rev_auto_corr);
/// assert_eq!(auto_corr as f32, 0.047983058);
/// ```
pub fn filter_autocorrelation(filter: &[f64], lag: isize) -> f64 {
    let lag : usize = lag.abs() as usize;

    if lag >= filter.len() {
        return 0.0
    }

    let mut rxx = 0.0;
    for i in lag..filter.len() {
        rxx += filter[i] * filter[i - lag];
    }

    rxx
}

// pub fn filter_autocorrelation_c(filter: &[f64])

/// Computes the cross-correlation of two filters at a specific lag
/// 
/// # Arguments
/// 
/// * `h` - the first filter
/// * `g` - the second filter
/// * `lag` - cross-correlation lag (samples)
/// 
/// # Example
/// 
/// ```
/// use solid::filter::firdes::{filter_crosscorrelation, firdes_kaiser, firdes_notch};
/// 
/// let h = match firdes_kaiser(51, 0.35, 120.0, 0.0) {
///     Ok(filter) => filter,
///     _ => vec!()
/// };
/// 
/// let g = match firdes_notch(25, 0.20, 30.0) {
///     Ok(filter) => filter,
///     _ => vec!()
/// };
/// 
/// let cross_corr = filter_crosscorrelation(&h, &g, 0);
/// 
/// assert_eq!(cross_corr as f32, 0.92825377);
/// ```
pub fn filter_crosscorrelation(h: &[f64], g: &[f64], lag: isize) -> f64 {
    if h.len() < g.len() {
        return filter_crosscorrelation(g, h, lag)
    }

    if lag <= -(g.len() as isize) { return 0.0 }
    if lag >= h.len() as isize { return 0.0 }

    let mut ig = 0;
    let mut ih = 0;

    if lag < 0 {
        ig = (-lag) as usize;
    }

    if lag > 0 {
        ih = lag as usize;
    }

    let n;
    if lag < 0 {
        n = g.len() as isize + lag;
    } else if lag < (h.len() - g.len()) as isize {
        n = g.len() as isize;
    } else {
        n = h.len() as isize - lag;
    }

    let mut rxy = 0.0;
    for i in 0..(n as usize) {
        rxy += h[ih + i] * g[ig + i];
    }

    rxy
}

/// Compute the inter-symbol interference (ISI) for both RMS and Max for the filter
/// 
/// The size of the filter should equal to (2 * sps * delay + 1)
/// 
/// # Arguments
/// 
/// * `filter` - the filter
/// * `samples_per_symbol` - filter over sampling rate
/// * `filter delay` - the delay of the filter in whole int symbols
/// 
/// # Example
/// 
/// ```
/// use solid::filter::firdes::{firdes_notch, filter_isi};
/// 
/// let h = match firdes_notch(25, 0.20, 30.0) {
///     Ok(filter) => filter,
///     _ => vec!()
/// };
/// 
/// let (rms, max) = filter_isi(&h, 1, 25);
/// 
/// assert_eq!(rms as f32, 0.02509764);
/// assert_eq!(max as f32, 0.061966006);
/// ```
pub fn filter_isi(filter: &[f64], samples_per_symbol: usize, filter_delay: usize) -> (f64, f64) {
    if 2 * samples_per_symbol * filter_delay + 1 != filter.len() {
        if cfg!(debug_assertion) {
            println!("Samples Per Symbol and Filter Delay do not match the filter's coefficient length!");
        }
        return (0.0, 0.0);
    }

    let rxx0 = filter_autocorrelation(filter, 0);

    let mut isi_rms = 0.0;
    let mut isi_max = 0.0;
    for i in 1..(2 * filter_delay) {
        let e = (filter_autocorrelation(filter, (i * samples_per_symbol) as isize) / rxx0).abs();
        isi_rms += e*e;

        if i == 1 || e > isi_max {
            isi_max = e;
        }
    }

    ((isi_rms / (2.0 * filter_delay as f64)).sqrt(), isi_max)
}

/// Compute Relative out-of-band energy
/// 
/// # Arguments
/// 
/// * `filter` the filter
/// * `cutoff_frequency` - the cutoff frequency to analyze at
/// * `fft_size` - size of the fft in bins
/// 
/// # Example
/// 
/// ```
/// use solid::filter::firdes::{firdes_notch, filter_energy};
/// 
/// let h = match firdes_notch(25, 0.20, 30.0) {
///     Ok(filter) => filter,
///     _ => vec!()
/// };
/// let energy = match filter_energy(&h, 0.35, 128) {
///     Ok(e) => e,
///     _ => 0.0
/// };
/// 
/// assert_eq!(energy as f32, 0.3152318);
/// ```
pub fn filter_energy(filter: &[f64], cutoff_frequency: f64, fft_size: usize) -> Result<f64, Box<dyn Error>> {
    if cutoff_frequency < 0.0 || cutoff_frequency > 0.5 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidBandwidth)))
    } else if filter.len() == 0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidFilterSize)))
    } else if fft_size == 0 {
        return Err(Box::new(FirdesError(FirdesErrorCode::InvalidFFTSize)))
    }

    let mut ejwt = vec![Complex::<f64>::new(0.0, 0.0); filter.len()];

    let mut e_total = 0.0;
    let mut e_stopband = 0.0;

    let dp = DotProduct::<f64>::new(filter.to_vec(), Direction::FORWARD);

    for i in 0..fft_size {
        let f = 0.5 * i as f64 / fft_size as f64;

        for k in 0..filter.len() {
            ejwt[k] = (Complex::new(0.0, 1.0) * 2.0 * PI_64 * f * k as f64).exp();
        }

        let v = Execute::execute(&dp, &ejwt);

        let e2 = (v * v.conj()).re;

        e_total += e2;
        if f > cutoff_frequency {
            e_stopband += e2;
        }
    }

    Ok(e_stopband / e_total)
}