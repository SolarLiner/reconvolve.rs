use convolve::convolution;

pub fn golay_sequence(power: u32) -> (Vec<i32>, Vec<i32>) {
    if power == 0 {
        (vec![1], vec![1])
    } else {
        let (a, b) = golay_sequence(power - 1);
        return (vec![a.clone(),b.clone()].concat(), vec![a, b.iter().copied().map(|x| -x).collect()].concat())
    }
}

pub fn deconvolve(signal_a: Vec<f64>, res_a: Vec<f64>, signal_b: Vec<f64>, res_b: Vec<f64>) -> Vec<f64> {
    assert_eq!(signal_a.len(), signal_b.len());
    assert_eq!(res_a.len(), res_b.len());
    let signal_len = signal_a.len();
    let vec_a = convolution(signal_a.into_iter().rev().collect(), res_a.to_vec());
    let vec_b = convolution(signal_b.into_iter().rev().collect(), res_b.to_vec());
    return vec_a.into_iter().zip(vec_b.into_iter()).map(|(a,b)| (a+b)/(2.0 * signal_len as f64)).collect();
}

#[cfg(test)]
mod tests {
    use super::golay_sequence;

    #[test]
    fn test_golay_simple() {
        let (a, b) = golay_sequence(0);
        assert_eq!(vec![1], a);
        assert_eq!(vec![1], b);
    }

    #[test]
    fn test_golay_next() {
        let (a, b) = golay_sequence(2);
        assert_eq!(vec![1,1,1,-1], a);
        assert_eq!(vec![1,1,-1,1], b);
    }
}