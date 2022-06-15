extern crate solid;

use solid::group_delay::*;


#[test]
fn delay_test() {
    let m: usize = 12;
    let fc: f32 = 0.25;

    let hlen: usize = 2 * m + 1;
    let h = vec![0.0; hlen];
    // let g = vec![0.0; hlen];

    for i in 0..hlen {
        let t = i as f32 - m as f32;
        h[i] = (fc * t).sin() * hamming(i, hlen);
    }

    let dt: f32 = -2.0;
    
    let dt_actual = fir_group_delay(&h, dt);
}