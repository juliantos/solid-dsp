use super::super::fft::{DataUnion, FFTDirection, FFTFlags, FFTMethod, FFTType, MixedRadix, FFT};
use super::super::resources::factor;

use std::mem::ManuallyDrop;

use num::Zero;
use num::complex::Complex;

fn estimate_mixed_radix(nfft: usize) -> usize {
    let factors = factor(nfft);

    let factors_len = factors.len();
    if factors_len < 2 {
        return 0;
    }

    let mut num_factors_2 = 0;
    for (i, &j) in factors.iter().enumerate() {
        num_factors_2 = i;
        if j != 2 {
            break;
        }
    }

    if num_factors_2 > 0 {
        if nfft % 16 == 0 {
            return 16;
        } else if nfft % 8 == 0 {
            return 8;
        } else if nfft % 4 == 0 {
            return 4;
        } else {
            return 2;
        }
    }

    factors[0]
}

pub fn create_mixed_radix_plan(nfft: usize, direction: FFTDirection, flags: FFTFlags) -> FFT {
    let q = estimate_mixed_radix(nfft);
    if q == 0 {
        panic!("Tried To Create a Mixed Radix Plan with a Prime nFFT {nfft}!")
    };
    if nfft % q != 0 {
        panic!("Tried To Create a Mixed Radix Plan Where nFFT: {nfft} is not divisible by {q}")
    };

    let p = nfft / q;
    let d = if direction == FFTDirection::FORWARD {
        -1.0
    } else {
        1.0
    };
    let mut twiddle = Vec::new();
    for i in 0..nfft {
        let exponent = Complex::from_polar(
            1.0,
            d * 2.0 * std::f64::consts::PI * (i as f64) / (nfft as f64),
        );
        twiddle.push(exponent);
    }

    FFT {
        nfft,
        fft_type: if direction == FFTDirection::FORWARD {
            FFTType::FORWARD
        } else {
            FFTType::REVERSE
        },
        fft_direction: direction,
        fft_method: FFTMethod::MIXEDRADIX,
        fft_flags: flags,
        data: DataUnion {
            mixed_radix: ManuallyDrop::new(MixedRadix {
                q,
                p,
                p_fft: Box::new(FFT::new(p, direction, flags)),
                q_fft: Box::new(FFT::new(q, direction, flags)),
                twiddle,
                execute_function: mixed_radix_execute,
            }),
        },
    }
}

pub fn mixed_radix_execute(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    let mut output = vec![Complex::<f64>::zero(); fft.nfft];

    let p_len = unsafe { fft.data.mixed_radix.p };
    let q_len = unsafe { fft.data.mixed_radix.q };

    let mut p_vec = vec![Complex::<f64>::zero(); p_len];
    let mut q_vec = vec![Complex::<f64>::zero(); q_len];

    let p_fft = unsafe { &fft.data.mixed_radix.p_fft };
    let q_fft = unsafe { &fft.data.mixed_radix.q_fft };
    let twiddle = unsafe { &fft.data.mixed_radix.twiddle };

    let mut intermediate_input = Vec::from(input);

    for i in 0..q_len {
        for j in 0..p_len {
            p_vec[j] = input[q_len * j + i];
        }

        let t = p_fft.execute(&p_vec)?;

        for j in 0..p_len {
            intermediate_input[q_len * j + i] = t[j] * twiddle[i * j];
        }
    }

    for i in 0..p_len {
        for j in 0..q_len {
            q_vec[j] = intermediate_input[q_len * i + j];
        }

        let t = q_fft.execute(&q_vec)?;

        for j in 0..q_len {
            output[p_len * j + i] = t[j];
        }
    }

    Ok(output)
}
