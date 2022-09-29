/// Polynomials
use super::ComplexSqrt;

use std::error::Error;
use std::fmt;

use num::complex::Complex;

const ITERATIONS: usize = 32;
const TOLERANCE: f64 = 1e-16;

#[derive(Debug)]
enum PolynomialErrorCode {
    InvalidOrder,
    IrreduciblePolynomial,
    InvalidPolynomialLength,
    FailedToConverge,
}

#[derive(Debug)]
struct PolynomialError(PolynomialErrorCode);

impl fmt::Display for PolynomialError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Polynomial Error: {:?}", self.0)
    }
}

impl Error for PolynomialError {}

/// Finds the Complex Roots of the Polynomial
///
/// # Arguments
///
/// * `polynomials` - The polynomial array, ascending powers
///
/// # Example
///
/// ```
/// use solid::math::poly::find_roots;
/// use num::complex::Complex;
///
/// let polynomial = [6.0, 11.0, -33.0, -33.0, 11.0, 6.0];
///
/// let roots = find_roots(&polynomial).unwrap();
///
/// let output = vec![Complex::new(-3.0, 0.0), Complex::new(-1.0, 0.0), Complex::new(-1.0/3.0, 0.0), Complex::new(0.5, 0.0), Complex::new(2.0, 0.0)];
/// assert_eq!(roots, output);
/// ```
pub fn find_roots(polynomials: &[f64]) -> Result<Vec<Complex<f64>>, Box<dyn Error>> {
    let mut roots = find_roots_bairstow(polynomials)?;

    roots.sort_by(|a, b| {
        let ar = a.re;
        let br = b.re;

        let ai = a.im;
        let bi = b.im;

        if ar == br {
            if ai > bi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        } else if ar > br {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    });

    Ok(roots)
}

/// Finds the Complex Roots of the Polynomial using Bairstow
///
/// # Arguments
///
/// * `polynomials` - The polynomial array, ascending powers
///
/// # Example
///
/// ```
/// use solid::math::poly::find_roots_bairstow;
/// use num::complex::Complex;
///
/// let polynomial = [6.0, 11.0, -33.0, -33.0, 11.0, 6.0];
///
/// let roots = find_roots_bairstow(&polynomial).unwrap();
///
/// let output = vec![Complex::new(-1.0/3.0, 0.0), Complex::new(-1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(-3.0, 0.0), Complex::new(0.5, 0.0)];
/// assert_eq!(roots, output);
/// ```
pub fn find_roots_bairstow(polynomials: &[f64]) -> Result<Vec<Complex<f64>>, Box<dyn Error>> {
    let mut input_poly: Vec<f64> = polynomials.to_vec();
    let mut output_poly: Vec<f64> = vec![];
    let mut roots: Vec<Complex<f64>> = vec![];

    let mut n = polynomials.len();
    if n == 0 {
        return Err(Box::new(PolynomialError(PolynomialErrorCode::InvalidOrder)));
    }

    let r = n % 2;
    let l = (n - r) / 2;
    let j = l - 1 + r;
    let mut last_i = 0;
    for i in 0..j {
        let mut u;
        let mut v;

        if i % 2 == 0 {
            if input_poly[n - 1] == 0.0 {
                return Err(Box::new(PolynomialError(
                    PolynomialErrorCode::IrreduciblePolynomial,
                )));
            }

            u = input_poly[n - 2] / input_poly[n - 1];
            v = input_poly[n - 3] / input_poly[n - 1];

            if n > 3 {
                (output_poly, u, v) = find_roots_bairstow_persistent(&input_poly, u, v)?;
            }
        } else {
            if output_poly[n - 1] == 0.0 {
                return Err(Box::new(PolynomialError(
                    PolynomialErrorCode::IrreduciblePolynomial,
                )));
            }

            u = output_poly[n - 2] / output_poly[n - 1];
            v = output_poly[n - 3] / output_poly[n - 1];

            if n > 3 {
                (input_poly, u, v) = find_roots_bairstow_persistent(&output_poly, u, v)?;
            }
        }

        let root = (u * u - 4.0 * v).csqrt();
        let complex_root_0 = 0.5 * (-u + root);
        let complex_root_1 = 0.5 * (-u - root);

        roots.push(complex_root_0);
        roots.push(complex_root_1);

        n -= 2;
        last_i = i;
    }

    if r == 0 {
        if last_i % 2 == 0 {
            roots.push(Complex::new(-output_poly[0] / output_poly[1], 0.0));
        } else {
            roots.push(Complex::new(-input_poly[0] / input_poly[1], 0.0));
        }
    }

    Ok(roots)
}

/// Runs multiple iterations of Bairstow's method finding quadratic factor x^2 + u*x + v
///
/// # Arguments
///
/// * `polynomials` - The polynomial array
/// * `u_estimate` - The u estimate
/// * `v_estimate` - The v estimate
///
/// # Example
///
/// ```
/// use solid::math::poly::find_roots_bairstow_recursion;
/// use num::complex::Complex;
///
/// let polynomial = [6.0, -9.0, -9.0, 6.0];
///
/// let roots = find_roots_bairstow_recursion(&polynomial, -1.5, -1.5).unwrap();
///
/// let output = vec![-3.0, 6.0];
/// assert_eq!(roots, (output, -1.0, -2.0));
/// ```
pub fn find_roots_bairstow_recursion(
    polynomials: &[f64],
    u_estimate: f64,
    v_estimate: f64,
) -> Result<(Vec<f64>, f64, f64), Box<dyn Error>> {
    if polynomials.len() < 3 {
        return Err(Box::new(PolynomialError(
            PolynomialErrorCode::InvalidPolynomialLength,
        )));
    }

    let mut u = u_estimate;
    let mut v = v_estimate;

    let n = polynomials.len() - 1;
    let mut iterations = 0;

    let mut b = vec![0.0; n + 1];
    let mut f = vec![0.0; n + 1];

    while iterations != ITERATIONS {
        iterations += 1;

        for i in (0..=(n - 2)).rev() {
            b[i] = polynomials[i + 2] - u * b[i + 1] - v * b[i + 2];
            f[i] = b[i + 2] - u * f[i + 1] - v * f[i + 2];
        }
        let c = polynomials[1] - u * b[0] - v * b[1];
        let g = b[1] - u * f[0] - v * f[1];
        let d = polynomials[0] - v * b[0];
        let h = b[0] - v * f[0];

        let q0 = v * g * g;
        let q1 = h * (h - u * g);
        let metric = (q0 + q1).abs();
        let q = if metric < TOLERANCE {
            u *= 0.5;
            v *= 0.5;
            continue;
        } else {
            1.0 / (q0 + q1)
        };

        let du = -q * (-h * c + g * d);
        let dv = -q * (-g * v * c + (g * u - h) * d);

        let step = du.abs() + dv.abs();

        u += du;
        v += dv;

        if step < TOLERANCE {
            break;
        }
    }

    let mut reduced_polynomials = vec![0.0; n - 1];
    reduced_polynomials[..(n - 1)].copy_from_slice(&b[..(n - 1)]);

    if iterations == ITERATIONS {
        return Err(Box::new(PolynomialError(
            PolynomialErrorCode::FailedToConverge,
        )));
    }

    Ok((reduced_polynomials, u, v))
}

/// Runs multiple iterations of Bairstow's method with different starting conditions
/// and looks for convergence
///
/// # Arguments
///
/// * `polynomials` - The polynomial array
/// * `u_estimate` - The u estimate
/// * `v_estimate` - The v estimate
///
/// # Example
///
/// ```
/// use solid::math::poly::find_roots_bairstow_persistent;
/// use num::complex::Complex;
///
/// let polynomial = [6.0, 11.0, -33.0, -33.0, 11.0, 6.0];
///
/// let roots = find_roots_bairstow_persistent(&polynomial, 1.8333333333333333333333, -5.5).unwrap();
///
/// let output = vec![18.0, -39.0, 3.0, 6.0];
/// assert_eq!(roots, (output, 4.0 / 3.0, 1.0 / 3.0));
/// ```
pub fn find_roots_bairstow_persistent(
    polynomials: &[f64],
    u_estimate: f64,
    v_estimate: f64,
) -> Result<(Vec<f64>, f64, f64), Box<dyn Error>> {
    let mut u = u_estimate;
    let mut v = v_estimate;
    for i in 0..ITERATIONS {
        match find_roots_bairstow_recursion(polynomials, u, v) {
            Ok(thing) => return Ok(thing),
            _ => {
                let val = (i as f64 * 1.1).cos() * (i as f64 * 0.2).exp();
                u = val;
                v = val;
            }
        }
    }

    Err(Box::new(PolynomialError(
        PolynomialErrorCode::FailedToConverge,
    )))
}

/// Expands the Binomial P_n(x) = (1 + x)^n as P_n(x) = p\[0\]*x + p\[1\]*x + p\[2\]*x^2 + ... + p\[n\]*x^n
///
/// # Arguments
///
/// * `roots` - The Power of the binomial
///
/// # Example
///
/// ```
/// use solid::math::poly::expand_binomial;
///
/// let polynomial = expand_binomial(5);
/// assert_eq!(polynomial, vec![1.0, 5.0, 10.0, 10.0, 5.0, 1.0]);
///
/// ```
pub fn expand_binomial(roots: usize) -> Vec<f64> {
    let mut output = Vec::new();

    if roots == 0 {
        output.push(0.0);
        return output;
    }

    output.push(1.0);
    let mut zeros = vec![0.0f64; roots];
    output.append(&mut zeros);

    for i in 0..roots {
        for j in (1..=(i + 1)).rev() {
            output[j] += output[j - 1];
        }
    }
    output
}

/// Expands the Binomial P_n(x) = (1 + x)^m * (1 - x)^k as
/// P_n(x) = p\[0\]*x + p\[1\]*x + p\[2\]*x^2 + ... + p\[n\]*x^n
///
/// # Arguments
///
/// * `m_roots` - M roots
/// * `k_roots` - K roots
///
/// # Example
///
/// ```
/// use solid::math::poly::expand_binomial_pm;
///
/// let polynomial = expand_binomial_pm(4, 3);
/// assert_eq!(polynomial, vec![1.0, 1.0, -3.0, -3.0, 3.0, 3.0, -1.0, -1.0]);
/// ```
pub fn expand_binomial_pm(m_roots: usize, k_roots: usize) -> Vec<f64> {
    let roots = m_roots + k_roots;
    let mut output = Vec::new();

    if roots == 0 {
        output.push(0.0);
    }

    output.push(1.0);
    let mut zeros = vec![0.0; roots];
    output.append(&mut zeros);

    for i in 0..m_roots {
        for j in (1..=(i + 1)).rev() {
            output[j] += output[j - 1];
        }
    }

    for i in m_roots..roots {
        for j in (1..=(i + 1)).rev() {
            output[j] -= output[j - 1];
        }
    }

    output
}
