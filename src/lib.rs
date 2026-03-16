use js_sys::Float64Array;
use wasm_bindgen::prelude::*;
// use web_sys::console::log_1;

#[wasm_bindgen]
pub struct OptimizerResult {
    x: Vec<f64>,
    pub f: f64,
    pub iterations: u32,
}

#[wasm_bindgen]
impl OptimizerResult {
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> Float64Array {
        Float64Array::from(&self.x[..])
    }
}

#[wasm_bindgen]
pub struct MaximumSharpeOptions {
    mu: Vec<f64>,
    pub rfr: f64,
}

#[wasm_bindgen]
impl MaximumSharpeOptions {
    #[wasm_bindgen(constructor)]
    pub fn new(mu: Vec<f64>, rfr: f64) -> MaximumSharpeOptions {
        MaximumSharpeOptions { mu, rfr }
    }

    #[wasm_bindgen(getter)]
    pub fn mu(&self) -> Float64Array {
        Float64Array::from(&self.mu[..])
    }
}

#[wasm_bindgen]
pub enum OptimizationMode {
    MinimumVariance = 0,
    MaximumSharpe = 1,
}

fn simplex_projection(v: &[f64]) -> Vec<f64> {
    let _n = v.len();
    let mut u = v.to_vec();
    u.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());

    let mut cssv = 0.0_f64;
    let mut rho = 0usize;
    for (i, &ui) in u.iter().enumerate() {
        cssv += ui;
        if ui > (cssv - 1.0) / (i + 1) as f64 {
            rho = i;
        }
    }

    let cssv_rho: f64 = u[..=rho].iter().sum();
    let theta = (cssv_rho - 1.0) / (rho + 1) as f64;

    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

fn excess_projection(y: &[f64], excess_returns: &[f64]) -> Vec<f64> {
    let y_pos: Vec<f64> = y
        .iter()
        .zip(excess_returns)
        .map(|(&yi, &ei)| if ei > 0.0 { yi.max(0.0) } else { 0.0 })
        .collect();
    let dot: f64 = y_pos
        .iter()
        .zip(excess_returns)
        .map(|(yi, ei)| yi * ei)
        .sum();

    if dot <= 0.0 {
        let n_positive = excess_returns.iter().filter(|&&e| e > 0.0).count();
        if n_positive == 0 {
            let n = y.len();
            return vec![1.0 / n as f64; n];
        }
        excess_returns
            .iter()
            .map(|&e| {
                if e > 0.0 {
                    1.0 / n_positive as f64
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        y_pos.iter().map(|yi| yi / dot).collect()
    }
}

fn portfolio_variance(x: &[f64], sigma: &[f64], n: usize) -> f64 {
    let mut sigma_x = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            sigma_x[i] += sigma[i * n + j] * x[j];
        }
    }

    x.iter().zip(&sigma_x).map(|(xi, sx)| 0.5 * xi * sx).sum()
}

fn portfolio_gradient(x: &[f64], sigma: &[f64], n: usize) -> Vec<f64> {
    let mut grad = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            grad[i] += sigma[i * n + j] * x[j];
        }
    }
    grad
}

fn min_variance(sigma_flat: &[f64], n: usize) -> OptimizerResult {
    const MAX_ITER: u32 = 2000;
    const TOL: f64 = 1e-8;

    const ALPHA: f64 = 0.3;
    const BETA: f64 = 0.5;
    const STEP_INIT: f64 = 1.0;

    let mut x = simplex_projection(&vec![1.0 / n as f64; n]);
    let mut iter = 0u32;

    for i in 0..MAX_ITER {
        iter = i + 1;
        let grad = portfolio_gradient(&x, &sigma_flat, n);

        let mut step = STEP_INIT;
        let f_curr = portfolio_variance(&x, &sigma_flat, n);
        let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();

        let x_new = loop {
            let candidate: Vec<f64> = x.iter().zip(&grad).map(|(xi, gi)| xi - step * gi).collect();
            let x_proj = simplex_projection(&candidate);
            let f_new = portfolio_variance(&x_proj, &sigma_flat, n);

            if f_new <= f_curr - ALPHA * step * grad_norm_sq {
                break x_proj;
            }
            step *= BETA;

            if step < 1e-15 {
                break x_proj;
            }
        };

        let delta: f64 = x_new
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        x = x_new;

        if delta < TOL {
            break;
        }
    }

    let f = portfolio_variance(&x, &sigma_flat, n);
    OptimizerResult {
        x,
        f,
        iterations: iter,
    }
}

fn max_sharpe(sigma_flat: &[f64], mu: &[f64], rfr: f64, n: usize) -> OptimizerResult {
    const MAX_ITER: u32 = 2000;
    const TOL: f64 = 1e-8;

    const BETA: f64 = 0.5;
    const STEP_INIT: f64 = 1.0;

    let excess_returns: Vec<f64> = mu.iter().map(|&m| m - rfr).collect();

    let uniform = vec![1.0 / n as f64; n];
    let mut y = excess_projection(&uniform, &excess_returns);
    // web_sys::console::log_1(
    //     &format!(
    //         "[RUST_WASM_BINARY]\ninitial y: {:?}, dot: {}",
    //         y,
    //         y.iter()
    //             .zip(&excess_returns)
    //             .map(|(yi, ei)| yi * ei)
    //             .sum::<f64>()
    //     )
    //     .into(),
    // );

    let mut iter = 0u32;

    for i in 0..MAX_ITER {
        iter = i + 1;
        let grad = portfolio_gradient(&y, sigma_flat, n);
        let f_curr = portfolio_variance(&y, sigma_flat, n);

        let mut step = STEP_INIT;
        let y_new = loop {
            let candidate: Vec<f64> = y.iter().zip(&grad).map(|(yi, gi)| yi - step * gi).collect();
            let y_proj = excess_projection(&candidate, &excess_returns);
            let f_new = portfolio_variance(&y_proj, sigma_flat, n);

            if f_new <= f_curr {
                break y_proj;
            }
            step *= BETA;
            if step < 1e-15 {
                break y_proj;
            }
        };

        let delta: f64 = y_new
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        y = y_new;
        if delta < TOL {
            break;
        }
    }

    let sum_y: f64 = y.iter().sum();
    let x: Vec<f64> = y.iter().map(|yi| yi / sum_y).collect();
    let f = portfolio_variance(&x, sigma_flat, n);

    OptimizerResult {
        x,
        f,
        iterations: iter,
    }
}

#[wasm_bindgen]
pub fn optimize(
    sigma_flat: &[f64],
    n: usize,
    mode: OptimizationMode,
    sharpe_options: Option<MaximumSharpeOptions>,
) -> OptimizerResult {
    // log_1(
    //     &format!(
    //         "[RUST_WASM_BINARY]\nsigma_flat.len(): {}\nn: {}",
    //         &sigma_flat.len(),
    //         n
    //     )
    //     .into(),
    // );

    match mode {
        OptimizationMode::MinimumVariance => return min_variance(sigma_flat, n),
        OptimizationMode::MaximumSharpe => match sharpe_options {
            Some(s_opts) => return max_sharpe(sigma_flat, &s_opts.mu, s_opts.rfr, n),
            None => OptimizerResult {
                x: vec![],
                f: -1.0,
                iterations: 0,
            },
        },
    }
}
