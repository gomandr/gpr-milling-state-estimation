# Gaussian process regression for quality estimation of milling operation

This semester project investigates the use of Gaussian Process Regression methods to estimate the quality of a part after milling operations. 

The milling process is studied to understand the key parameters at play. This provides knowledge on which timesteps bring relevant information to increase the prediction performances at a given time. It also confirms from a theoretical viewpoint that power and vibration measurements extracted from the simulation are meaningful for the prediction problem.

The results obtained show the importance of the different preprocessing steps implemented. Cleaning of the dataset and data-augmentation with previous timesteps lead to
increase of the predictive performances of the models trained. Feature selection doesn’t improve the results but significantly reduces the computation time. This indicates that for time critical predictions this step should not be neglected. 

The kernels used in this project are the most common found in the literature due to limited prior knowledge. Further research on multi-task learning and the relationship between the milling parameters and the quality estimates should therefore be carried out. This would allow the use of Gaussian Processes at their full potential by constructing kernels shaped specifically for the milling operations. For real-time applications, approximate methods with uncertain inputs should be explored to overcome the limitations of exact Gaussian Processes.
