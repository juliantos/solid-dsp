pub mod complex;

use std::f64::consts::PI as PI_64;
use std::f32::consts::PI as PI_32;

pub trait Sinc<T> {
    /// Computes  sinc(x) = sin(pi/x) / (pi/x)
    fn sinc(&self) -> T;
}

impl Sinc<f64> for f64 {
    #[inline]
    fn sinc(&self) -> f64 {
        if self.abs() < 0.01 {
            return (PI_64 * self / 2.0).cos() * (PI_64 * self / 4.0).cos() * (PI_64 * self / 8.0).cos()
        }
        (PI_64 * self).sin() / (PI_64 * self)
    }
}

impl Sinc<f32> for f32 {
    #[inline]
    fn sinc(&self) -> f32 {
        if self.abs() < 0.01 {
            return (PI_32 * self / 2.0).cos() * (PI_32 * self / 4.0).cos() * (PI_32 * self / 8.0).cos()
        }
        (PI_32 * self).sin() / (PI_32 * self)
    }
}

pub trait Bessel<T> {
    /// Modified Bessel Function of the first kind
    fn besseli(self, nu: T) -> T;

    /// Log( Bessel ) Function
    fn lnbesseli(self, nu: T) -> T; 

    // Bessel Function of the first kind
    fn besselj(self, nu: T) -> T;
}

impl Bessel<f64> for f64 {
    fn besseli(self, nu: f64) -> f64 {
        // Special case self == 0
        if self == 0.0 {
            if nu == 0.0 {
                return 1.0
            } else {
                return 0.0
            }
        }

        // Special case nu == 1/2
        if nu == 0.5 {
            return (2.0 / (PI_64 * self)).sqrt() * self.sinh()
        } 

        // Low signal approximation
        if self < 0.001 * (nu + 1.0).sqrt() {
            return (0.5 * self).powf(nu) / (nu + 1.0).gamma()
        }

        // Otherwise derive from log expression
        self. lnbesseli(nu).exp()
    }

    fn lnbesseli(self, nu: f64) -> f64 {
        // Special case self == 0
        if self == 0.0 {
            if nu == 0.0  {
                return 0.0
            } else {
                return -f64::MAX;
            }
        }

        // Special case nu == 1/2
        if nu == 0.5 {
            return 0.5 * (2.0 / (PI_64 * self)).ln() + self.sinh().ln()
        }

        // Low Signal Approximation
        if self < 0.001 * (nu + 1.0).sqrt() {
            return - (nu + 1.0).gamma() + nu * (0.5 * self).ln()
        }

        // Otherwise
        let t0 = nu * (0.5 * self).ln();
        let mut y = 0.0;

        // 64 is the number of BESSEL Iterations
        // FIXME: should be configurable by compiler
        for k in 0..64 {
            // compute log( (z^2/4)^k )
            let t1 = 2.0 * k as f64 * (0.5 * self).ln();
            // compute: log( k! * Gamma(nu + k +1) )
            let t2 = (k as f64 + 1.0).lngamma();
            let t3 = (nu + k as f64 + 1.0).lngamma();
            // accumulate y
            y += (t1 - t2 - t3).exp();
        }
        
        t0 + y.ln()
    }

    fn besselj(self, nu: f64) -> f64 {
        // Special case self == 0
        if self == 0.0 {
            if nu == 0.0 {
                return 1.0
            } else {
                return 0.0
            }
        }

        if self < 0.001 * (nu + 1.0).sqrt() {
            return (0.5 * self).powf(nu) / (nu + 1.0).gamma()
        }

        #[allow(non_snake_case)]
        let mut J = 0.0;

        let abs_nu = nu.abs();

        // FIXME: Bessel J Iterations
        for i in 0..128 {
            // compute: (2i + |nu|)
            let t0 = 2.0 * i as f64 + abs_nu;

            // compute: (2i + |nu|)*log(z)
            let t1 = t0 * self.ln();

            // compute: (2i + |nu|)*log(2)
            let t2 = t0 * (2f64).ln();

            // compute: log(Gamma(i+1))
            let t3 = (i as f64 + 1.0).lngamma();

            // compute: log(Gamma(|nu|+i+1))
            let t4 = (abs_nu + i as f64 + 1.0).lngamma();

            if i % 2 == 0 {
                J += (t1 - t2 - t3 - t4).exp();
            } else {
                J -= (t1 - t2 - t3 - t4).exp();
            }
        }

        J
    }
}

impl Bessel<f32> for f32 {
    fn besseli(self, nu: f32) -> f32 {
        // Special case self == 0
        if self == 0.0 {
            if nu == 0.0 {
                return 1.0
            } else {
                return 0.0
            }
        }

        // Special case nu == 1/2
        if nu == 0.5 {
            return (2.0 / (PI_32 * self)).sqrt() * self.sinh()
        } 

        // Low signal approximation
        if self < 0.001 * (nu + 1.0).sqrt() {
            return (0.5 * self).powf(nu) / (nu + 1.0).gamma()
        }

        // Otherwise derive from log expression
        self. lnbesseli(nu).exp()
    }

    fn lnbesseli(self, nu: f32) -> f32 {
        // Special case self == 0
        if self == 0.0 {
            if nu == 0.0  {
                return 0.0
            } else {
                return -f32::MAX;
            }
        }

        // Special case nu == 1/2
        if nu == 0.5 {
            return 0.5 * (2.0 / (PI_32 * self)).ln() + self.sinh().ln()
        }

        // Low Signal Approximation
        if self < 0.001 * (nu + 1.0).sqrt() {
            return - (nu + 1.0).gamma() + nu * (0.5 * self).ln()
        }

        // Otherwise
        let t0 = nu * (0.5 * self).ln();
        let mut y = 0.0;

        // 64 is the number of BESSEL Iterations
        // FIXME: should be configurable by compiler
        for k in 0..64 {
            // compute log( (z^2/4)^k )
            let t1 = 2.0 * k as f32 * (0.5 * self).ln();
            // compute: log( k! * Gamma(nu + k +1) )
            let t2 = (k as f32 + 1.0).gamma();
            let t3 = (nu + k as f32 + 1.0).gamma();
            // accumulate y
            y += (t1 - t2 - t3).exp();
        }
        
        t0 + y.ln()
    }

    fn besselj(self, nu: f32) -> f32 {
        // Special case self == 0
        if self == 0.0 {
            if nu == 0.0 {
                return 1.0
            } else {
                return 0.0
            }
        }

        if self < 0.001 * (nu + 1.0).sqrt() {
            return (0.5 * self).powf(nu) / (nu + 1.0).gamma()
        }

        #[allow(non_snake_case)]
        let mut J = 0.0;

        let abs_nu = nu.abs();

        // FIXME: Bessel J Iterations
        for i in 0..128 {
            // compute: (2i + |nu|)
            let t0 = 2.0 * i as f32 + abs_nu;

            // compute: (2i + |nu|)*log(z)
            let t1 = t0 * self.ln();

            // compute: (2i + |nu|)*log(2)
            let t2 = t0 * (2f32).ln();

            // compute: log(Gamma(i+1))
            let t3 = (i as f32 + 1.0).lngamma();

            // compute: log(Gamma(|nu|+i+1))
            let t4 = (abs_nu + i as f32 + 1.0).lngamma();

            if i % 2 == 0 {
                J += (t1 - t2 - t3 - t4).exp();
            } else {
                J -= (t1 - t2 - t3 - t4).exp();
            }
        }

        J
    }
}

pub trait Gamma<T> {
    /// Gamma Function 
    fn gamma(self) -> T;
    /// Log( Gamma ) Function
    fn lngamma(self) -> T;
}

impl Gamma<f64> for f64 {
    fn gamma(self) -> f64 {
        if self < 0.0 {
            let t0 = (1.0 - self).gamma();
            let t1 = (PI_64 * self).sin();

            if t0 == 0.0 || t1 == 0.0 {
                if cfg!(debug_assertion) {
                    panic!("Gamma Divides By Zero");
                }
            }

            return PI_64 / (t0 * t1)
        } else {
            self.lngamma().exp()
        }
    }

    fn lngamma(self) -> f64 {
        if self < 0.0 {
            if cfg!(debug_assertion) {
                panic!("Log Gamma Undefined for self < 0")
            }
            return 0.0   
        } else if self < 10.0 {
            (self + 1.0).lngamma() - self.ln()
        } else {
            let g = 0.5 * ((2.0 * PI_64).ln() - self.ln());
            let h = g + self * ((self + (1.0 / (12.0 * self - 0.1 / self))).ln() - 1.0);
            h
        }
    }
}

impl Gamma<f32> for f32 {
    fn gamma(self) -> f32 {
        if self < 0.0 {
            let t0 = (1.0 - self).gamma();
            let t1 = (PI_32 * self).sin();

            if t0 == 0.0 || t1 == 0.0 {
                if cfg!(debug_assertion) {
                    panic!("Gamma Divides By Zero");
                }
            }

            return PI_32 / (t0 * t1)
        } else {
            self.lngamma().exp()
        }
    }

    fn lngamma(self) -> f32 {
        if self < 0.0 {
            if cfg!(debug_assertion) {
                panic!("Log Gamma Undefined for self < 0")
            }
            return 0.0   
        } else if self < 10.0 {
            (self + 1.0).lngamma() - self.ln()
        } else {
            let g = 0.5 * ((2.0 * PI_32).ln() - self.ln());
            let h = g + self * ((self + (1.0 / (12.0 * self - 0.1 / self))).ln() - 1.0);
            h
        }
    }
}