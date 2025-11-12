The script 'fso_ch_est_2' uses FSO signal datasets to predict the attenuation that will occur in the signal over specified 'latency' time horizons ahead.
To achieve the forward-looking prediction, features are created based on data samples lagged according to the latency values.
The data is also made stationary by first-order differencing.
The predicted attenuation is used to pre-compensate the original signal to reduce the distortions in the signal.

The Rytov_variance quantifies the distortion present in the signal by fitting the signal's distribution against known distributions (gamma-gamma and log-normal).
This enables the determination of the Rytov value that minimizes the fitting.
