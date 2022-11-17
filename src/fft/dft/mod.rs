use super::super::dot_product::{Direction, DotProduct};
use super::super::fft::{
    DataUnion, Dft, FFTDirection, FFTError, FFTErrorCode, FFTExecuteFunction, FFTFlags, FFTMethod,
    FFTType, FFT,
};

use std::mem::ManuallyDrop;

use num::Zero;
use num::complex::Complex;

const SQRT_3_2: f64 = 0.866025403784439;
const G: Complex<f64> = Complex::new(-0.5, -SQRT_3_2);
const GI: Complex<f64> = Complex::new(-0.5, SQRT_3_2);
const G0: Complex<f64> = Complex::new(0.309016994374947, -0.951056516295154);
const G1: Complex<f64> = Complex::new(-0.809016994374947, -0.587785252292473);
const G0I: Complex<f64> = Complex::new(0.309016994374947, 0.951056516295154);
const G1I: Complex<f64> = Complex::new(-0.809016994374947, 0.587785252292473);
const G2: Complex<f64> = Complex::new(0.623489801858734, -0.781831482468030);
const G3: Complex<f64> = Complex::new(-0.222520933956314, -0.974927912181824);
const G4: Complex<f64> = Complex::new(-0.900968867902419, -0.433883739117558);
const G5: Complex<f64> = Complex::new(
    std::f64::consts::FRAC_1_SQRT_2,
    -std::f64::consts::FRAC_1_SQRT_2,
);
const G6: Complex<f64> = Complex::new(
    std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
);
const G7: Complex<f64> = Complex::new(
    -std::f64::consts::FRAC_1_SQRT_2,
    -std::f64::consts::FRAC_1_SQRT_2,
);
const G8: Complex<f64> = Complex::new(
    -std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
);
const G9: Complex<f64> = Complex::new(0.92387950, -0.38268346);
const G10: Complex<f64> = Complex::new(0.38268343, -0.92387950);
const G11: Complex<f64> = Complex::new(-0.38268343, -0.92387950);
const G12: Complex<f64> = Complex::new(-0.92387950, -0.38268346);
const G13: Complex<f64> = Complex::new(0.92387950, 0.38268346);
const G14: Complex<f64> = Complex::new(0.38268343, 0.92387950);
const G15: Complex<f64> = Complex::new(-0.38268343, 0.92387950);
const G16: Complex<f64> = Complex::new(-0.92387950, 0.38268346);

pub fn create_dft_plan(nfft: usize, direction: FFTDirection, flags: FFTFlags) -> FFT {
    let mut twiddle: Vec<Complex<f64>> = Vec::new();
    let mut dot_products: Vec<DotProduct<Complex<f64>>> = Vec::new();
    let execute_fn: FFTExecuteFunction;
    match nfft {
        2 => {
            execute_fn = dft_execute2;
        }
        3 => {
            execute_fn = dft_execute3;
        }
        4 => {
            execute_fn = dft_execute4;
        }
        5 => {
            execute_fn = dft_execute5;
        }
        6 => {
            execute_fn = dft_execute6;
        }
        7 => {
            execute_fn = dft_execute7;
        }
        8 => {
            execute_fn = dft_execute8;
        }
        16 => {
            execute_fn = dft_execute16;
        }
        _ => {
            execute_fn = dft_execute;

            let d = if direction == FFTDirection::FORWARD {
                -1.0
            } else {
                1.0
            };
            twiddle = vec![Complex::<f64>::zero(); nfft];

            for i in 0..nfft {
                for j in 1..nfft {
                    let exponent = Complex::from_polar(
                        1.0,
                        d * 2.0 * std::f64::consts::PI * ((i * j) as f64) / (nfft as f64),
                    );
                    twiddle[j - 1] = exponent;
                }

                dot_products.push(DotProduct::new(&twiddle[0..(nfft - 1)], Direction::FORWARD));
            }
        }
    }

    FFT {
        nfft,
        fft_type: if direction == FFTDirection::FORWARD {
            FFTType::FORWARD
        } else {
            FFTType::REVERSE
        },
        fft_direction: direction,
        fft_method: FFTMethod::DFT,
        fft_flags: flags,
        data: DataUnion {
            dft: ManuallyDrop::new(Dft {
                twiddle,
                dot_products,
                execute_function: execute_fn,
            }),
        },
    }
}

pub fn dft_execute(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    let mut output = Vec::new();

    for i in 0..fft.nfft {
        let y = unsafe { fft.data.dft.dot_products[i].execute(&input[1..]) } + input[0];
        output.push(y);
    }

    Ok(output)
}

pub fn dft_execute2(
    _fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    if input.len() < 2 {
        return Err(Box::new(FFTError(FFTErrorCode::NotEnoughBuffer)));
    }

    let mut output = vec![Complex::<f64>::zero(); 2];
    output[0] = input[0] + input[1];
    output[1] = input[0] - input[1];

    Ok(output)
}

pub fn dft_execute3(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    if input.len() < 3 {
        return Err(Box::new(FFTError(FFTErrorCode::NotEnoughBuffer)));
    }

    let mut output = vec![Complex::<f64>::zero(); 3];

    output[0] = input[0] + input[1] + input[2];
    let ta = input[0] + input[1] * G + input[2] * GI;
    let tb = input[0] + input[1] * GI + input[2] * G;

    match fft.fft_direction {
        FFTDirection::FORWARD => {
            output[1] = ta;
            output[2] = tb;
        }
        FFTDirection::REVERSE => {
            output[1] = tb;
            output[2] = ta;
        }
    }

    Ok(output)
}

pub fn dft_execute4(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    if input.len() < 4 {
        return Err(Box::new(FFTError(FFTErrorCode::NotEnoughBuffer)));
    }

    let mut output = vec![Complex::<f64>::zero(); 4];

    // Reversal
    output[0] = input[0];
    output[1] = input[2];
    output[2] = input[1];
    output[3] = input[3];

    let tmp = output[1];
    output[1] = output[0] - tmp;
    output[0] += tmp;

    let tmp = output[3];
    output[3] = output[2] - tmp;
    output[2] += tmp;

    let tmp = output[2];
    output[2] = output[0] - tmp;
    output[0] += tmp;

    let tmp = Complex::new(output[3].im, -output[3].re);
    if fft.fft_direction == FFTDirection::FORWARD {
        output[3] = output[1] - tmp;
        output[1] += tmp;
    } else {
        output[3] = output[1] + tmp;
        output[1] -= tmp;
    }

    Ok(output)
}

pub fn dft_execute5(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    if input.len() < 5 {
        return Err(Box::new(FFTError(FFTErrorCode::NotEnoughBuffer)));
    }

    let mut output = vec![Complex::<f64>::zero(); 5];

    output[0] = input[0..5].iter().sum();

    match fft.fft_direction {
        FFTDirection::FORWARD => {
            output[1] = input[0] + input[1] * G0 + input[2] * G1 + input[3] * G1I + input[4] * G0I;
            output[2] = input[0] + input[1] * G1 + input[2] * G0I + input[3] * G0 + input[4] * G1I;
            output[3] = input[0] + input[1] * G1I + input[2] * G0 + input[3] * G0I + input[4] * G1;
            output[4] = input[0] + input[1] * G0I + input[2] * G1I + input[3] * G1 + input[4] * G0;
        }
        FFTDirection::REVERSE => {
            output[1] = input[0] + input[1] * G0I + input[2] * G1I + input[3] * G1 + input[4] * G0;
            output[2] = input[0] + input[1] * G1I + input[2] * G0 + input[3] * G0I + input[4] * G1;
            output[3] = input[0] + input[1] * G1 + input[2] * G0I + input[3] * G0 + input[4] * G1I;
            output[4] = input[0] + input[1] * G0 + input[2] * G1 + input[3] * G1I + input[4] * G0I;
        }
    }

    Ok(output)
}

pub fn dft_execute6(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    if input.len() < 6 {
        return Err(Box::new(FFTError(FFTErrorCode::NotEnoughBuffer)));
    }

    let mut output = vec![Complex::<f64>::zero(); 6];

    output[0] = input[0..6].iter().sum();

    let g1;
    let g2;
    let g3;
    let g4;
    match fft.fft_direction {
        FFTDirection::FORWARD => {
            g1 = -GI;
            g2 = G;
            g3 = GI;
            g4 = -G;
        }
        FFTDirection::REVERSE => {
            g1 = -G;
            g2 = GI;
            g3 = G;
            g4 = -GI;
        }
    }

    output[1] = input[0] + input[1] * g1 + input[2] * g2 - input[3] + input[4] * g3 + input[5] * g4;
    output[2] = input[0] + input[1] * g2 + input[2] * g3 + input[3] + input[4] * g2 + input[5] * g3;
    output[3] = input[0] - input[1] + input[2] - input[3] + input[4] - input[5];
    output[4] = input[0] + input[1] * g3 + input[2] * g2 + input[3] + input[4] * g3 + input[5] * g2;
    output[5] = input[0] + input[1] * g4 + input[2] * g3 - input[3] + input[4] * g2 + input[5] * g1;

    Ok(output)
}

pub fn dft_execute7(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    if input.len() < 7 {
        return Err(Box::new(FFTError(FFTErrorCode::NotEnoughBuffer)));
    }

    let mut output = vec![Complex::<f64>::zero(); 7];

    output[0] = input[0..7].iter().sum();

    let g1;
    let g2;
    let g3;
    match fft.fft_direction {
        FFTDirection::FORWARD => {
            g1 = G2;
            g2 = G3;
            g3 = G4;
        }
        FFTDirection::REVERSE => {
            g1 = G2.conj();
            g2 = G3.conj();
            g3 = G4.conj();
        }
    }

    let g4 = g3.conj();
    let g5 = g2.conj();
    let g6 = g1.conj();

    output[1] = input[0]
        + input[1] * g1
        + input[2] * g2
        + input[3] * g3
        + input[4] * g4
        + input[5] * g5
        + input[6] * g6;
    output[2] = input[0]
        + input[1] * g2
        + input[2] * g4
        + input[3] * g6
        + input[4] * g1
        + input[5] * g3
        + input[6] * g5;
    output[3] = input[0]
        + input[1] * g3
        + input[2] * g6
        + input[3] * g2
        + input[4] * g5
        + input[5] * g1
        + input[6] * g4;
    output[4] = input[0]
        + input[1] * g4
        + input[2] * g1
        + input[3] * g5
        + input[4] * g2
        + input[5] * g6
        + input[6] * g3;
    output[5] = input[0]
        + input[1] * g5
        + input[2] * g3
        + input[3] * g1
        + input[4] * g6
        + input[5] * g4
        + input[6] * g2;
    output[6] = input[0]
        + input[1] * g6
        + input[2] * g5
        + input[3] * g4
        + input[4] * g3
        + input[5] * g2
        + input[6] * g1;

    Ok(output)
}

pub fn dft_execute8(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    if input.len() < 8 {
        return Err(Box::new(FFTError(FFTErrorCode::NotEnoughBuffer)));
    }

    let mut output = vec![Complex::<f64>::zero(); 8];

    output[0] = input[0];
    output[1] = input[4];
    output[2] = input[2];
    output[3] = input[6];
    output[4] = input[1];
    output[5] = input[5];
    output[6] = input[3];
    output[7] = input[7];

    let mut yp = output[1];
    output[1] = output[0] - yp;
    output[0] += yp;
    yp = output[3];
    output[3] = output[2] - yp;
    output[2] += yp;
    yp = output[5];
    output[5] = output[4] - yp;
    output[4] += yp;
    yp = output[7];
    output[7] = output[6] - yp;
    output[6] += yp;
    yp = output[2];
    output[2] = output[0] - yp;
    output[0] += yp;
    yp = output[6];
    output[6] = output[4] - yp;
    output[4] += yp;

    let mut yp1;
    match fft.fft_direction {
        FFTDirection::FORWARD => {
            yp = Complex::new(output[3].im, -output[3].re);
            yp1 = Complex::new(output[7].im, -output[7].re);
        }
        FFTDirection::REVERSE => {
            yp = Complex::new(-output[3].im, output[3].re);
            yp1 = Complex::new(-output[7].im, output[7].re);
        }
    }
    output[3] = output[1] - yp;
    output[1] += yp;
    output[7] = output[5] - yp1;
    output[5] += yp1;
    yp = output[4];
    output[4] = output[0] - yp;
    output[0] += yp;

    let yp2;
    match fft.fft_direction {
        FFTDirection::FORWARD => {
            yp = output[5] * G5;
            yp1 = Complex::new(output[6].im, -output[6].re);
            yp2 = output[7] * G7;
        }
        FFTDirection::REVERSE => {
            yp = output[5] * G6;
            yp1 = Complex::new(-output[6].im, output[6].re);
            yp2 = output[7] * G8;
        }
    }
    output[5] = output[1] - yp;
    output[1] += yp;
    output[6] = output[2] - yp1;
    output[2] += yp1;
    output[7] = output[3] - yp2;
    output[3] += yp2;

    Ok(output)
}

pub fn dft_execute16(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    if input.len() < 16 {
        return Err(Box::new(FFTError(FFTErrorCode::NotEnoughBuffer)));
    }

    let mut output = vec![Complex::<f64>::zero(); 16];

    output[0] = input[0];
    output[1] = input[8];
    output[2] = input[4];
    output[3] = input[12];
    output[4] = input[2];
    output[5] = input[10];
    output[6] = input[6];
    output[7] = input[14];
    output[8] = input[1];
    output[9] = input[9];
    output[10] = input[5];
    output[11] = input[13];
    output[12] = input[3];
    output[13] = input[11];
    output[14] = input[7];
    output[15] = input[15];

    // i=0
    let mut yp = output[1];
    output[1] = output[0] - yp;
    output[0] += yp;
    yp = output[3];
    output[3] = output[2] - yp;
    output[2] += yp;
    yp = output[5];
    output[5] = output[4] - yp;
    output[4] += yp;
    yp = output[7];
    output[7] = output[6] - yp;
    output[6] += yp;
    yp = output[9];
    output[9] = output[8] - yp;
    output[8] += yp;
    yp = output[11];
    output[11] = output[10] - yp;
    output[10] += yp;
    yp = output[13];
    output[13] = output[12] - yp;
    output[12] += yp;
    yp = output[15];
    output[15] = output[14] - yp;
    output[14] += yp;

    // i=1
    yp = output[2];
    output[2] = output[0] - yp;
    output[0] += yp;
    yp = output[6];
    output[6] = output[4] - yp;
    output[4] += yp;
    yp = output[10];
    output[10] = output[8] - yp;
    output[8] += yp;
    yp = output[14];
    output[14] = output[12] - yp;
    output[12] += yp;

    match fft.fft_direction {
        FFTDirection::FORWARD => {
            yp = -output[3] * Complex::new(0.0, 1.0);
            output[3] = output[1] - yp;
            output[1] += yp;
            yp = -output[7] * Complex::new(0.0, 1.0);
            output[7] = output[5] - yp;
            output[5] += yp;
            yp = -output[11] * Complex::new(0.0, 1.0);
            output[11] = output[9] - yp;
            output[9] += yp;
            yp = -output[15] * Complex::new(0.0, 1.0);
            output[15] = output[13] - yp;
            output[13] += yp;
        }
        FFTDirection::REVERSE => {
            yp = output[3] * Complex::new(0.0, 1.0);
            output[3] = output[1] - yp;
            output[1] += yp;
            yp = output[7] * Complex::new(0.0, 1.0);
            output[7] = output[5] - yp;
            output[5] += yp;
            yp = output[11] * Complex::new(0.0, 1.0);
            output[11] = output[9] - yp;
            output[9] += yp;
            yp = output[15] * Complex::new(0.0, 1.0);
            output[15] = output[13] - yp;
            output[13] += yp;
        }
    }

    yp = output[4];
    output[4] = output[0] - yp;
    output[0] += yp;
    yp = output[12];
    output[12] = output[8] - yp;
    output[8] += yp;
    match fft.fft_direction {
        FFTDirection::FORWARD => {
            yp = output[5] * G5;
            output[5] = output[1] - yp;
            output[1] += yp;
            yp = output[13] * G5;
            output[13] = output[9] - yp;
            output[9] += yp;
            yp = -output[6] * Complex::new(0.0, 1.0);
            output[6] = output[2] - yp;
            output[2] += yp;
            yp = -output[14] * Complex::new(0.0, 1.0);
            output[14] = output[10] - yp;
            output[10] += yp;
            yp = output[7] * G7;
            output[7] = output[3] - yp;
            output[3] += yp;
            yp = output[15] * G7;
            output[15] = output[11] - yp;
            output[11] += yp;
        }
        FFTDirection::REVERSE => {
            yp = output[5] * G6;
            output[5] = output[1] - yp;
            output[1] += yp;
            yp = output[13] * G6;
            output[13] = output[9] - yp;
            output[9] += yp;
            yp = output[6] * Complex::new(0.0, 1.0);
            output[6] = output[2] - yp;
            output[2] += yp;
            yp = output[14] * Complex::new(0.0, 1.0);
            output[14] = output[10] - yp;
            output[10] += yp;
            yp = output[7] * G8;
            output[7] = output[3] - yp;
            output[3] += yp;
            yp = output[15] * G8;
            output[15] = output[11] - yp;
            output[11] += yp;
        }
    }

    yp = output[8];
    output[8] = output[0] - yp;
    output[0] += yp;
    match fft.fft_direction {
        FFTDirection::FORWARD => {
            yp = output[9] * G9;
            output[9] = output[1] - yp;
            output[1] += yp;
            yp = output[10] * G5;
            output[10] = output[2] - yp;
            output[2] += yp;
            yp = output[11] * G10;
            output[11] = output[3] - yp;
            output[3] += yp;
            yp = -output[12] * Complex::new(0.0, 1.0);
            output[12] = output[4] - yp;
            output[4] += yp;
            yp = output[13] * G11;
            output[13] = output[5] - yp;
            output[5] += yp;
            yp = output[14] * G7;
            output[14] = output[6] - yp;
            output[6] += yp;
            yp = output[15] * G12;
            output[15] = output[7] - yp;
            output[7] += yp;
        }
        FFTDirection::REVERSE => {
            yp = output[9] * G13;
            output[9] = output[1] - yp;
            output[1] += yp;
            yp = output[10] * G6;
            output[10] = output[2] - yp;
            output[2] += yp;
            yp = output[11] * G14;
            output[11] = output[3] - yp;
            output[3] += yp;
            yp = output[12] * Complex::new(0.0, 1.0);
            output[12] = output[4] - yp;
            output[4] += yp;
            yp = output[13] * G15;
            output[13] = output[5] - yp;
            output[5] += yp;
            yp = output[14] * G8;
            output[14] = output[6] - yp;
            output[6] += yp;
            yp = output[15] * G16;
            output[15] = output[7] - yp;
            output[7] += yp;
        }
    }

    Ok(output)
}
