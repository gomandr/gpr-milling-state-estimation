import gpytorch

class Matern_Kernel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nu=2.5, dims='one'):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.nu = nu
        self.dims = dims
        
        if self.dims == 'one':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=self.nu))
        elif self.dims == 'all':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=self.nu, \
                                                                                           ard_num_dims=train_x.shape[1]))                   
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
class SE_Kernel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dims='one'):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.dims = dims
        
        if self.dims == 'one':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif self.dims == 'all':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))          
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
class RQ_Kernel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dims='one'):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.dims = dims
        
        if self.dims == 'one':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        elif self.dims == 'all':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=train_x.shape[1]))          
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
class RQxLinear_Kernel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dims='one'):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.dims = dims
              
        if self.dims == 'one':
            self.covar_module = gpytorch.kernels.LinearKernel(active_dims=[0, train_x.shape[1]-1]) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        elif self.dims == 'all':
            self.covar_module = gpytorch.kernels.LinearKernel(active_dims=[0, train_x.shape[1]-1]) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=train_x.shape[1]))          
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
       
class PERxRQaddSE_Kernel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        PER_covar = gpytorch.kernels.CosineKernel()
        SE_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))
        RQ_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=train_x.shape[1]))          
            
        self.covar_module = PER_covar * (SE_covar + RQ_covar)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)