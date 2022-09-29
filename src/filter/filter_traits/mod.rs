use super::super::group_delay::*;
use super::super::math::complex::{Conj, Real};
use super::fir_filter::FIRFilter;
use super::iir_filter::second_order_filter::SecondOrderFilter;
use super::iir_filter::{IIRFilter, IIRFilterType};

use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg};

use num::complex::Complex;
use num_traits::Num;

pub trait Filter<C> {
    type Float;
    type Complex;

    /// Computes the Complex Frequency response of the filter
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use solid::filter::filter_traits::Filter;
    /// use solid::filter::firdes::*;
    /// use num::complex::Complex;
    ///
    /// let coefs = match firdes_notch(25, 0.35, 120.0) {
    ///     Ok(coefs) => coefs,
    ///     _ => vec!()
    /// };
    /// let filter = FIRFilter::<f64, f64>::new(&coefs, 1.0).unwrap();
    /// let response = Filter::frequency_response(&filter, 0.0);
    ///
    /// assert_eq!(response.re.round(), 1.0);
    /// assert_eq!(response.im, 0.0);
    /// ```
    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex;

    /// Computes the group delay in samples
    ///
    /// # Example
    ///
    /// ```
    /// use solid::filter::fir_filter::FIRFilter;
    /// use solid::filter::filter_traits::Filter;
    /// use solid::filter::firdes;
    ///
    /// let coefs = match firdes::firdes_notch(12, 0.35, 120.0) {
    ///     Ok(coefs) => coefs,
    ///     _ => vec!()
    /// };
    /// let filter = FIRFilter::<f64, f64>::new(&coefs, 1.0).unwrap();
    /// let delay = Filter::group_delay(&filter, 0.0);
    ///
    /// assert_eq!((delay + 0.5) as usize, 12);
    /// ```
    fn group_delay(&self, frequency: Self::Float) -> f64;
}

/// Implementation where the Coefficients are not a Complex Type
impl<C: Num + Copy + Sum, T: Copy> Filter<C> for FIRFilter<C, T>
where
    Complex<C>: Sum<Complex<C>>,
    Complex<C>: Add<Complex<f64>, Output = Complex<C>>,
    Complex<C>: AddAssign<Complex<f64>>,
    C: Mul<f64, Output = f64>,
    C: Mul<Complex<f64>, Output = Complex<f64>>,
    C: PartialOrd<f64>,
{
    type Float = C;
    type Complex = Complex<C>;

    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex {
        let mut output: Self::Complex = Complex::new(Self::Float::zero(), Self::Float::zero());

        let coefs = self.coefficients();
        for (i, coef) in coefs.iter().enumerate() {
            let out = *coef
                * Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
            output += out;
        }
        output * self.get_scale()
    }

    fn group_delay(&self, frequency: Self::Float) -> f64
    where
        C: PartialOrd<f64>,
    {
        match fir_group_delay(self.coefficients(), frequency) {
            Ok(delay) => delay,
            Err(e) => {
                if cfg!(debug_assertions) {
                    println!("{}", e);
                }
                0.0
            }
        }
    }
}

/// Implementation where Coefficients are Complex Type
impl<C: Num + Copy + Sum, T: Copy> Filter<C> for FIRFilter<Complex<C>, T>
where
    Complex<C>: Sum<Complex<C>>,
    Complex<C>: Mul<Complex<f64>, Output = Complex<f64>>,
    Complex<C>: Mul<f64, Output = Complex<C>>,
    Complex<C>: Add<Complex<f64>, Output = Complex<C>>,
    Complex<C>: AddAssign<Complex<f64>>,
    C: Mul<f64, Output = f64>,
    C: PartialOrd<f64>,
{
    type Float = C;
    type Complex = Complex<C>;

    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex {
        let mut output: Self::Complex = Complex::new(Self::Float::zero(), Self::Float::zero());

        let coefs = self.coefficients();
        for (i, coef) in coefs.iter().enumerate() {
            let out = *coef
                * Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
            output += out;
        }

        output * self.get_scale()
    }

    fn group_delay(&self, frequency: Self::Float) -> f64 {
        match fir_group_delay(self.coefficients(), frequency) {
            Ok(delay) => delay,
            Err(e) => {
                if cfg!(debug_assertions) {
                    println!("{}", e);
                }
                0.0
            }
        }
    }
}

impl<C: Num + Copy + Sum, T: Copy> Filter<C> for SecondOrderFilter<C, T>
where
    Complex<C>: Add<Complex<f64>, Output = Complex<C>>,
    C: Mul<f64, Output = f64>,
    C: Mul<Complex<f64>, Output = Complex<f64>>,
    C: Real<Output = C>,
    C: PartialOrd<f64>,
    C: Conj<Output = C>,
{
    type Float = C;
    type Complex = Complex<C>;

    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex {
        let mut output_b: Self::Complex = Complex::new(Self::Float::zero(), Self::Float::zero());
        let mut output_a: Self::Complex = Complex::new(Self::Float::zero(), Self::Float::zero());

        let coefs_b = self.numerator_coefs();
        let coefs_a = self.denominator_coefs();
        for i in 0..3 {
            let polar =
                Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
            if i < 2 {
                let out_b = coefs_b[i] * polar;
                output_b = output_b + out_b;
            }
            let out_a = coefs_a[i] * polar;
            output_a = output_a + out_a;
        }

        output_b / output_a
    }

    fn group_delay(&self, frequency: Self::Float) -> f64 {
        let mut coefs_b = Vec::<C>::new();
        let mut coefs_a = Vec::<C>::new();

        for i in 0..3 {
            if i == 2 {
                coefs_b.push(Self::Float::zero());
            } else {
                coefs_b.push(self.numerator_coefs()[i].real());
            }
            coefs_a.push(self.denominator_coefs()[i].real());
        }

        match iir_group_delay(&coefs_b, &coefs_a, frequency) {
            Ok(delay) => delay + 2.0,
            Err(e) => {
                if cfg!(debug_assertions) {
                    println!("{}", e);
                }
                0.0
            }
        }
    }
}

impl<C: Num + Copy + Sum, T: Copy> Filter<C> for SecondOrderFilter<Complex<C>, T>
where
    Complex<C>: Add<Complex<f64>, Output = Complex<C>>,
    Complex<C>: Mul<Complex<f64>, Output = Complex<f64>>,
    C: Mul<Complex<f64>, Output = Complex<f64>>,
    C: Mul<f64, Output = f64>,
    C: Real<Output = C>,
    C: PartialOrd<f64>,
    C: Conj<Output = C>,
{
    type Float = C;
    type Complex = Complex<C>;

    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex {
        let mut output_b: Self::Complex = Complex::new(Self::Float::zero(), Self::Float::zero());
        let mut output_a: Self::Complex = Complex::new(Self::Float::zero(), Self::Float::zero());

        let coefs_b = self.numerator_coefs();
        let coefs_a = self.denominator_coefs();
        for i in 0..3 {
            let polar =
                Complex::from_polar(1.0, frequency * 2.0 * std::f64::consts::PI * (i as f64));
            let out_b = coefs_b[i] * polar;
            let out_a = coefs_a[i] * polar;
            output_b = output_b + out_b;
            output_a = output_a + out_a;
        }

        output_b / output_a
    }

    fn group_delay(&self, frequency: Self::Float) -> f64 {
        let mut coefs_b = Vec::<C>::new();
        let mut coefs_a = Vec::<C>::new();

        for i in 0..3 {
            coefs_b.push(self.numerator_coefs()[i].real());
            coefs_a.push(self.denominator_coefs()[i].real());
        }

        match iir_group_delay(&coefs_b, &coefs_a, frequency) {
            Ok(delay) => delay + 2.0,
            Err(e) => {
                if cfg!(debug_assertions) {
                    println!("{}", e);
                }
                0.0
            }
        }
    }
}

impl<C: Num + Copy + Sum, T: Copy> Filter<C> for IIRFilter<C, T>
where
    Complex<C>: Add<Complex<f64>, Output = Complex<C>>,
    C: Mul<f64, Output = f64>,
    C: Mul<Complex<f64>, Output = Complex<f64>>,
    C: Real<Output = C>,
    C: PartialOrd<f64>,
    C: Conj<Output = C>,
{
    type Float = C;
    type Complex = Complex<C>;

    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex {
        match self.iir_type() {
            IIRFilterType::Normal => {
                let mut b = Complex::new(Self::Float::zero(), Self::Float::zero());
                let mut a = Complex::new(Self::Float::zero(), Self::Float::zero());

                for (i, &coef) in self.numerator_coefs().iter().enumerate() {
                    let exp = coef
                        * Complex::from_polar(
                            1.0,
                            frequency * 2.0 * std::f64::consts::PI * (i as f64),
                        );
                    b = b + exp;
                }

                for (i, &coef) in self.denominator_coefs().iter().enumerate() {
                    let exp = coef
                        * Complex::from_polar(
                            1.0,
                            frequency * 2.0 * std::f64::consts::PI * (i as f64),
                        );
                    a = a + exp;
                }

                b / a
            }
            IIRFilterType::SecondOrder => {
                let mut h = Complex::new(Self::Float::one(), Self::Float::zero());

                for filter in self.second_order_filters() {
                    h = h * Filter::frequency_response(filter, frequency);
                }

                h
            }
        }
    }

    fn group_delay(&self, frequency: Self::Float) -> f64 {
        match self.iir_type() {
            IIRFilterType::Normal => {
                match iir_group_delay(self.numerator_coefs(), self.denominator_coefs(), frequency) {
                    Ok(delay) => delay,
                    Err(e) => {
                        if cfg!(debug_assertions) {
                            println!("{}", e);
                        }
                        0.0
                    }
                }
            }
            IIRFilterType::SecondOrder => {
                let mut delay = 0.0;
                for filter in self.second_order_filters().iter() {
                    delay = delay + Filter::group_delay(filter, frequency) + 2.0;
                }
                delay
            }
        }
    }
}

impl<C: Num + Copy + Sum, T: Copy> Filter<C> for IIRFilter<Complex<C>, T>
where
    Complex<C>: Add<Complex<f64>, Output = Complex<C>>,
    Complex<C>: Mul<Complex<f64>, Output = Complex<f64>>,
    Complex<C>: Mul<Complex<C>, Output = Complex<C>>,
    C: Mul<Complex<f64>, Output = Complex<f64>>,
    C: Mul<f64, Output = f64>,
    C: Real<Output = C>,
    C: PartialOrd<f64>,
    C: Conj<Output = C>,
    C: Neg<Output = C>,
{
    type Float = C;
    type Complex = Complex<C>;

    fn frequency_response(&self, frequency: Self::Float) -> Self::Complex {
        match self.iir_type() {
            IIRFilterType::Normal => {
                let mut b = Complex::new(Self::Float::zero(), Self::Float::zero());
                let mut a = Complex::new(Self::Float::zero(), Self::Float::zero());

                for (i, &coef) in self.numerator_coefs().iter().enumerate() {
                    let exp = coef
                        * Complex::from_polar(
                            1.0,
                            frequency * 2.0 * std::f64::consts::PI * (i as f64),
                        );
                    b = b + exp;
                }

                for (i, &coef) in self.denominator_coefs().iter().enumerate() {
                    let exp = coef
                        * Complex::from_polar(
                            1.0,
                            frequency * 2.0 * std::f64::consts::PI * (i as f64),
                        );
                    a = a + exp;
                }

                b / a
            }
            IIRFilterType::SecondOrder => {
                let mut h = Complex::new(Self::Float::zero(), Self::Float::zero());

                for filter in self.second_order_filters() {
                    h = h * Filter::frequency_response(filter, frequency);
                }

                h
            }
        }
    }

    fn group_delay(&self, frequency: Self::Float) -> f64 {
        match self.iir_type() {
            IIRFilterType::Normal => {
                match iir_group_delay(self.numerator_coefs(), self.denominator_coefs(), frequency) {
                    Ok(delay) => delay,
                    Err(e) => {
                        if cfg!(debug_assertions) {
                            println!("{}", e);
                        }
                        0.0
                    }
                }
            }
            IIRFilterType::SecondOrder => {
                let mut delay = 0.0;
                for filter in self.second_order_filters().iter() {
                    delay = delay + Filter::group_delay(filter, frequency) + 2.0;
                }
                delay
            }
        }
    }
}
