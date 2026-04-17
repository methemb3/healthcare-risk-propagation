# Healthcare Risk Propagation

A prototype implementation of iterative risk propagation across a 
patient network using eigenvector methods.

## The Problem
Traditional healthcare risk models score patients in isolation. This 
model treats patients as nodes in a network, propagating risk across 
shared relationships until the system reaches equilibrium.

## The Algorithm
The core update rule is:

r(t+1) = α · A · r(t) + (1 − α) · b

Where:
- `A` = row-normalized patient adjacency matrix
- `b` = baseline clinical risk vector
- `α` = propagation coefficient (default: 0.85)

The system converges to the dominant eigenvector of the network — 
a global equilibrium risk profile.

## Demo
The included demo runs on a synthetic 6-patient network to validate 
the algorithm. Edges represent shared risk relationships such as 
same neighborhood, similar conditions, or shared physician.

## Usage
```bash
pip install -r requirements.txt
python risk_propagation.py
```

## Author
Methembe Moses Ncube  

