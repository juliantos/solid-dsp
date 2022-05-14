//! Just Some Extraneous Functions


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