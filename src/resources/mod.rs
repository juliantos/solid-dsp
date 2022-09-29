//! Just Some Extraneous Functions

static MAX_FACTORS: usize = 64;

/// Gets the leading index bit location
///
/// # Example
///
/// ```
/// use solid::resources::msb_index;
/// let value = 0b1;
/// let index = msb_index(value);
/// assert_eq!(index, 1);
///
/// let value = 129;
/// let index = msb_index(value);
/// assert_eq!(index, 8);
///
/// ```
#[inline(always)]
pub fn msb_index(x: usize) -> usize {
    (usize::BITS - x.leading_zeros()) as usize
}

/// Gets the factors of the number
///
/// # Example
///
/// ```
/// use solid::resources::factor;
/// let value = 12;
/// let mut factors = factor(value);
/// factors.sort();
/// assert_eq!(factors, vec![2, 2, 3])
/// ```
#[inline(always)]
pub fn factor(number_to_factor: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut n = number_to_factor;
    while n > 1 && factors.len() < MAX_FACTORS {
        for i in 2..=n {
            if n % i == 0 {
                factors.push(i);
                n /= i;
                break;
            }
        }
    }

    factors
}

/// Computes (Base ^ Exponent) % Modulo
///
/// # Example
///
/// ```
/// use solid::resources::modpow;
/// let base = 5;
/// let exp = 5;
/// let n = 3;
/// let c = modpow(base, exp, n);
///
/// assert_eq!(c, 2);
/// ```
pub fn modpow(base: usize, exp: usize, n: usize) -> usize {
    let mut c = 1;
    for _ in 0..exp {
        c = (c * base) % n;
    }

    c
}

/// Gets the Primitive Prime Root of a number
///
/// # Example
///
/// ```
/// use solid::resources::primitive_root_prime;
/// let value = 43;
/// let prime = primitive_root_prime(value);
///
/// assert_eq!(prime, 3);
/// ```
pub fn primitive_root_prime(n: usize) -> usize {
    let mut factors = Vec::new();
    let mut n_ = n - 1;
    while n_ > 1 && factors.len() < MAX_FACTORS {
        for k in 2..=n_ {
            if n_ % k == 0 {
                if factors.iter().find(|&&x| x == k) == None {
                    factors.push(k)
                }
                n_ /= k;
                break;
            }
        }
    }

    let mut h = 0;
    for g in 2..n {
        h = g;
        let mut is_root = true;
        for item in &factors {
            let e = (n - 1) / item;
            if modpow(g, e, n) == 1 {
                is_root = false;
                break;
            }
        }

        if is_root {
            break;
        }
    }

    h
}
