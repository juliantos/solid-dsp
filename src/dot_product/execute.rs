pub trait Execute<I, O> {
    /// Computes the Dot Product of the Samples and the Coefficients in the DotProduct
    /// Object.
    ///
    /// # Example
    ///
    /// ```
    /// use solid::dot_product::{DotProduct, Direction, execute::Execute};
    ///
    /// let coefs = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let dp = DotProduct::new(&coefs, Direction::REVERSE);
    /// let mul = vec![1.0; 5];
    /// let exe = dp.execute(&mul);
    ///
    /// assert_eq!(exe, 15.0);
    /// ```
    fn execute(&self, samples: &[I]) -> O;
}