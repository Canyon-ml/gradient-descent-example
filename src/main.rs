
// Hyperparameters
/// How many times you want to regress
const EPOCH: i32 = 10;
/// The Learning Rate. 
/// Keeps the gradients sane. 
const LR: f32 = 0.001;

// dataset from the linreg notes
const DATASET: [(f32, f32); 13] = [
//    x    y
    (1.0, 2.0),
    (2.0, 4.0),
    (2.0, 5.0),
    (3.0, 3.0),
    (4.0, 4.0),
    (4.0, 5.0),
    (4.0, 6.0),
    (5.0, 7.0),
    (6.0, 5.0),
    (7.0, 7.0),
    (7.0, 8.0),
    (8.0, 7.0),
    (9.0, 10.0),
];

fn main() {
    // Initialize Slope and Base to any value
    let mut m = 1.5;
    let mut b = 1.5;

    for e in 0..EPOCH {
        // Store the Average Loss to be computed (for visualization)
        let mut avg_loss = 0.0_f32;
        // for every Input and Target in the Dataset
        for (x, yhat) in DATASET.iter() {
            // formula for a line, y = mx + b
            let y = m * x + b;

            // Compute the loss
            // Add to avg_loss
            avg_loss += yhat - y;

            // Compute the error
            // Derivative of loss function
            let delta = y - yhat;

            // Update The Slope according to the gradient with respect to M, which is X. 
            m -= delta * x * LR;
            // Update the Base according to the gradient with respect to B, which is 1. 
            b -= delta * LR;
        }
        println!("Epoch: {e}, Avg Loss: {}", avg_loss / DATASET.len() as f32);
    }
    println!("M: {m}, B: {b}");

    // For Linear Regression over any Function, we need to compute the partial derivative with respect to every Trainable Parameter, in this case M and B. 
}
