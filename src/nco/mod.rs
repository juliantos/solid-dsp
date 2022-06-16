
use std::fmt;
use std::error::Error;

use num::complex::Complex;

#[derive(Debug, PartialEq)]
pub enum NCOErrorCode {
    BandwidthOutOfRange
}

#[derive(Debug)]
pub struct NCOError(pub NCOErrorCode);

impl fmt::Display for NCOError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let error_code = match self.0 {
            NCOErrorCode::BandwidthOutOfRange => "Bandwidth out Range [0, inf)"
        };
        write!(f, "NCO Error {}", error_code)
    }
}

impl Error for NCOError {}

#[derive(Debug)]
pub struct NCO {
    look_up_table: [f64; 1024],
    theta: u32,
    delta_theta: u32,
    alpha: f64,
    beta: f64
}

impl NCO {
    pub fn new() -> Self {
        let mut table = [0.0; 1024];
        for i in 0..1024{
            table[i] = (2.0 * std::f64::consts::PI * (i as f64) / 1024.0).sin();
        }
        let a = 0.1f64;
        let b = a.sqrt();
        NCO {
            look_up_table: table,
            theta: 0,
            delta_theta: 0,
            alpha: a,
            beta: b
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.theta = 0;
        self.delta_theta = 0;
    }

    #[inline]
    pub fn set_frequency(&mut self, delta_theta: f64) {
        self.delta_theta = constrain(delta_theta);
    }

    #[inline]
    pub fn adjust_frequency(&mut self, dt: f64) {
        self.delta_theta += constrain(dt);
    }

    #[inline]
    pub fn get_frequency(&self) -> f64 {
        let dt = (self.delta_theta as u64 / 1u64 << 32) as f64 * 2.0f64 * std::f64::consts::PI;
        if dt > std::f64::consts::PI {
            dt - 2.0f64 * std::f64::consts::PI
        } else {
            dt
        }
    }

    #[inline]
    pub fn set_phase(&mut self, phi: f64) {
        self.theta = constrain(phi);
    }

    #[inline]
    pub fn adjust_phase(&mut self, delta_phi: f64) {
        self.theta += constrain(delta_phi)
    }

    #[inline]
    pub fn get_phase(&self) -> f64 {
        (self.theta as u64 / 1u64 << 32) as f64 * 2.0f64 * std::f64::consts::PI
    }

    #[inline]
    pub fn step(&mut self) {
        self.theta = self.theta.wrapping_add(self.delta_theta);
    }

    #[inline]
    fn index(&self) -> usize {
        (((self.theta + (1 << 21)) >> 22) & 0x3ff) as usize
    }

    #[inline]
    pub fn sin(&self) -> f64 {
        self.look_up_table[self.index()]
    }

    #[inline]
    pub fn cos(&self) -> f64 {
        let index = (self.index() + 256) & 0x3ff;
        self.look_up_table[index]
    }

    #[inline]
    pub fn sincos(&self) -> (f64, f64) {
        (self.sin(), self.cos())
    }

    #[inline]
    pub fn complex_exponential(&self) -> Complex<f64> {
        Complex::new(self.cos(), self.sin())
    }

    pub fn set_internal_pll_bandwidth(&mut self, bandwidth: f64) -> Result<(), Box<dyn Error>>{
        if bandwidth < 0.0 {
            return Err(Box::new(NCOError(NCOErrorCode::BandwidthOutOfRange)))
        }

        self.alpha = bandwidth;
        self.beta = self.alpha.sqrt();
        Ok(())
    }

    #[inline]
    pub fn pll_step(&mut self, delta_phi: f64) {
        self.adjust_frequency(delta_phi * self.alpha);
        self.adjust_phase(delta_phi * self.beta);
    }

    #[inline]
    pub fn mix_up(&self, input: Complex<f64>) -> Complex<f64> {
        let complex_phasor = self.complex_exponential();
        complex_phasor * input
    }

    #[inline]
    pub fn mix_down(&self, input: Complex<f64>) -> Complex<f64> {
        let complex_phasor = self.complex_exponential().conj();
        complex_phasor * input
    }

    #[inline]
    pub fn mix_up_block(&mut self, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut output = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            output[i] = self.mix_up(input[i]);
            self.step();
        }

        output
    }

    #[inline]
    pub fn mix_down_block(&mut self, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut output = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            output[i] = self.mix_down(input[i]);
            self.step();
        }

        output
    }   
}

pub fn constrain(theta: f64) -> u32 {
    // Divide the theta by 2PI
    let theta_div_two_pi = theta / (2.0 * std::f64::consts::PI);
    // Take only the fractional part
    let mut fractional_part = theta_div_two_pi.fract();
    
    // Make fractional part positive
    if fractional_part < 0.0 {
        fractional_part += 1.0;
    }

    (fractional_part * 0xffffffffu32 as f64) as u32
}

impl fmt::Display for NCO {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NCO [Theta={}] [ΔTheta={}] [Alpha={}] [Beta={}]", self.theta, self.delta_theta, self.alpha, self.beta)
    }
}