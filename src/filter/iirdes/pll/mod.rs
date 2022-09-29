//! Second Order Infinite Impulse Response Phase-Locked Loop Filter Design

use super::*;

use std::error::Error;

/// Design Second Order IIR Filter with active lag
///
/// # Arguments
///
/// * `bandwidth` - Bandwidth of the Desired filter
/// * `damping_factor` - Damping Factor recommended 1.0 / sqrt(2.0)
/// * `loop_gain` - Loop Gain recommened 1000.0
///
/// # Examples
///
/// ```
/// use solid::filter::*;
///
/// let filter = iirdes::pll::active_lag(0.35, 1.0 / (2.0f64).sqrt(), 1000.0).unwrap();
///
/// assert_eq!(filter.0[1], 4000.0);
/// ```
pub fn active_lag(
    bandwidth: f64,
    damping_factor: f64,
    loop_gain: f64,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    if bandwidth <= 0.0 {
        return Err(Box::new(IirdesError(IirdesErrorCode::Bandwidth)));
    } else if damping_factor <= 0.0 {
        return Err(Box::new(IirdesError(IirdesErrorCode::DampingFactor)));
    } else if loop_gain <= 0.0 {
        return Err(Box::new(IirdesError(IirdesErrorCode::Gain)));
    }

    let t1 = loop_gain / (bandwidth * bandwidth);
    let t2 = 2.0 * damping_factor / bandwidth - 1.0 / loop_gain;

    let mut num = vec![0.0; 3];
    let mut den = vec![0.0; 3];

    num[0] = 2.0 * loop_gain * (1.0 + t2 / 2.0);
    num[1] = 2.0 * loop_gain * 2.0;
    num[2] = 2.0 * loop_gain * (1.0 - t2 / 2.0);

    den[0] = 1.0 + t1 / 2.0;
    den[1] = -t1;
    den[2] = -1.0 + t1 / 2.0;

    Ok((num, den))
}

/// Design Second Order IIR Filter with PI
///
/// # Arguments
///
/// * `bandwidth` - Bandwidth of the Desired filter
/// * `damping_factor` - Damping Factor recommended 1.0 / sqrt(2.0)
/// * `loop_gain` - Loop Gain recommened 1000.0
///
/// # Examples
///
/// ```
/// use solid::filter::*;
///
/// let filter = iirdes::pll::active_proportional_integral(0.35, 1.0 / (2.0f64).sqrt(), 1000.0).unwrap();
///
/// assert_eq!(filter.0[1], 4000.0);
/// ```
pub fn active_proportional_integral(
    bandwidth: f64,
    damping_factor: f64,
    loop_gain: f64,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    if bandwidth <= 0.0 {
        return Err(Box::new(IirdesError(IirdesErrorCode::Bandwidth)));
    } else if damping_factor <= 0.0 {
        return Err(Box::new(IirdesError(IirdesErrorCode::DampingFactor)));
    } else if loop_gain <= 0.0 {
        return Err(Box::new(IirdesError(IirdesErrorCode::Gain)));
    }

    let t1 = loop_gain / (bandwidth * bandwidth);
    let t2 = 2.0 * damping_factor / bandwidth - 1.0 / loop_gain;

    let mut num = vec![0.0; 3];
    let mut den = vec![0.0; 3];

    num[0] = 2.0 * loop_gain * (1.0 + t2 / 2.0);
    num[1] = 2.0 * loop_gain * 2.0;
    num[2] = 2.0 * loop_gain * (1.0 - t2 / 2.0);

    den[0] = t1 / 2.0;
    den[1] = -t1;
    den[2] = t1 / 2.0;

    Ok((num, den))
}
