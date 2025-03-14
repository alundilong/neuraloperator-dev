"""
data_losses.py contains code to compute standard data objective 
functions for training Neural Operators. 

By default, losses expect arguments y_pred (model predictions) and y (ground y.)
"""

import math
from typing import List

import torch
import torch.distributed as dist

from neuralop.losses.finite_diff import central_diff_1d, central_diff_2d, central_diff_3d

#loss function with rel/abs Lp loss
class LpLoss(object):
    """
    LpLoss provides the L-p norm between two 
    discretized d-dimensional functions. Note ihat 
    LpLoss always averages over the spatial dimensions.

    .. note :: 
        In function space, the Lp norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to LpLoss, we assume a regular 
        discretization and take ``1 / measure`` as the quadrature weights. 

    Parameters
    ----------
    d : int, optional
        dimension of data on which to compute, by default 1
    p : int, optional
        order of L-norm, by default 2
        L-p norm: [\sum_{i=0}^n (x_i - y_i)**p] ** (1/p)
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        .. note::

        To perform quadrature, ``LpLoss`` scales ``measure`` by the size
        of each spatial dimension of ``x``, and multiplies them with 
        ||x-y||, such that the final norm is a scaled average over the spatial
        dimensions of ``x``. 
    reduction : str, optional
        whether to reduce across the batch and channel dimensions
        by summing ('sum') or averaging ('mean')

        .. warning:: 

            ``LpLoss`` always reduces over the spatial dimensions according to ``self.measure``.
            `reduction` only applies to the batch and channel dimensions.

    Examples
    --------

    ```
    """

    def __init__(self,
            d=1,
            p=2,
            measure=1.,
            reduction='sum',
            data_processor=None,
            loss_type='relative',
            mask_channel_outputs=None,
            mask_channel=None):
        super().__init__()

        self.d = d
        self.p = p
        self.data_processor = data_processor

        # Ensure loss_type is either a string or a list of strings
        if not (isinstance(loss_type, str) or (isinstance(loss_type, list) and all(isinstance(l, str) for l in loss_type))):
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Expected a string or a list of strings.")

        self.loss_type = loss_type
        self.mask_channel_outputs = mask_channel_outputs
        self.mask_channel = mask_channel
        
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that 
        ``LpLoss`` computes the average over spatial dims. 

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature

    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x)
        else:
            x = torch.mean(x)
        
        return x

    def reduce_channel(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x,dim=0)
        else:
            x = torch.mean(x,dim=0)
        
        return x

    def abs(self, x, y, quadrature=None, mask_tensor=None, mask_channel_outputs=None):
        """absolute Lp-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature weights for integral
            either single scalar or one per dimension
        """
    
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        const = math.prod(quadrature)**(1.0/self.p)

        # Compute absolute difference
        diff = torch.abs(x - y)  # Shape: (batch, channels, ...)
    
        # Apply mask if provided
        if mask_tensor is not None and mask_channel_outputs is not None:
            # Ensure mask_channel_outputs is a tensor for proper indexing
            if isinstance(mask_channel_outputs, list):
                mask_channel_outputs = torch.tensor(mask_channel_outputs, device=x.device)
            
            # Apply mask only on selected channels
            diff[:, mask_channel_outputs, ...] *= mask_tensor
        # Compute Lp norm of the masked difference
        diff_norm = torch.norm(torch.flatten(diff, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff_channel_wise = self.reduce_channel(diff_norm)

        diff = self.reduce_all(diff_norm).squeeze()
            
        return diff, diff_channel_wise

    def rel(self, x, y, mask_tensor=None, mask_channel_outputs=None):
        """
        rel: relative LpLoss
        Computes ||x - y|| / ||y||
    
        Parameters
        ----------
        x : torch.Tensor
            inputs (predicted values)
        y : torch.Tensor
            targets (ground truth)
        mask_tensor : torch.Tensor, optional
            Mask tensor to be applied only on specified channels.
        mask_channel_outputs : list or torch.Tensor, optional
            Indices of channels where the mask should be applied.
        """
    
        # Compute absolute difference
        diff = torch.abs(x - y)  # Shape: (batch, channels, ...)
    
        # Apply mask if provided
        if mask_tensor is not None and mask_channel_outputs is not None:
            # Ensure mask_channel_outputs is a tensor for proper indexing
            if isinstance(mask_channel_outputs, list):
                mask_channel_outputs = torch.tensor(mask_channel_outputs, device=x.device)
            
            # Apply mask only on selected channels
            diff[:, mask_channel_outputs, ...] *= mask_tensor
    
        # Compute Lp norm of the masked difference
        diff_norm = torch.norm(torch.flatten(diff, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)
    
        # Compute Lp norm of the target (denominator)
        y_norm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)
    
        # Avoid division by zero
        y_norm = torch.where(y_norm == 0, torch.tensor(1.0, device=y.device), y_norm)
    
        # Compute relative Lp loss
        diff = diff_norm / y_norm

        diff_channel_wise = self.reduce_channel(diff)

        diff = self.reduce_all(diff).squeeze()

        return diff, diff_channel_wise

    def hybrid(self, x, y, mask_tensor=None, mask_channel_outputs=None, loss_types = None):
        """
        rel: relative LpLoss
        Computes ||x - y|| / ||y||
    
        Parameters
        ----------
        x : torch.Tensor
            inputs (predicted values)
        y : torch.Tensor
            targets (ground truth)
        mask_tensor : torch.Tensor, optional
            Mask tensor to be applied only on specified channels.
        mask_channel_outputs : list or torch.Tensor, optional
            Indices of channels where the mask should be applied.
        """
    
        # Compute absolute difference
        diff = torch.abs(x - y)  # Shape: (batch, channels, ...)
    
        # Apply mask if provided
        if mask_tensor is not None and mask_channel_outputs is not None:
            # Ensure mask_channel_outputs is a tensor for proper indexing
            if isinstance(mask_channel_outputs, list):
                mask_channel_outputs = torch.tensor(mask_channel_outputs, device=x.device)
            
            # Apply mask only on selected channels
            diff[:, mask_channel_outputs, ...] *= mask_tensor
    
        # Compute Lp norm of the masked difference
        diff_norm = torch.norm(torch.flatten(diff, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)
    
        # Compute Lp norm of the target (denominator)
        y_norm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)
    
        # Avoid division by zero
        y_norm = torch.where(y_norm == 0, torch.tensor(1.0, device=y.device), y_norm)
    
        # Ensure loss_types has the correct length
        num_channels = diff_norm.shape[1]  # The second dimension is the channel dimension
        if len(loss_types) != num_channels:
            raise ValueError(f"Mismatch: loss_types has {len(loss_types)} elements, but expected {num_channels}.")
        
        diff = torch.zeros_like(diff_norm)
        # Iterate over channels and apply different loss calculations
        for channel in range(num_channels):
            if loss_types[channel] == "relative":
                diff[:, channel, ...] = diff_norm[:, channel, ...] / y_norm[:, channel, ...]
            else:
                diff[:, channel, ...] = diff_norm[:, channel, ...]  # Keep absolute error

        diff_channel_wise = self.reduce_channel(diff)

        diff = self.reduce_all(diff).squeeze()

        return diff, diff_channel_wise

    def __call__(self, y_pred, y, **kwargs):
        input_x = kwargs['x'].clone()
        mask_tensor = None
        if self.data_processor is not None:
            if self.data_processor.in_normalizer is not None:
                input_x = self.data_processor.in_normalizer.inverse_transform(input_x)
                mask_tensor = input_x[:,self.mask_channel:self.mask_channel+1,:,:,:]
        #print(mask_tensor.max(), mask_tensor.min(), mask_tensor.mean())
        if isinstance(self.loss_type, str):
            if self.loss_type == "relative":
                return self.rel(y_pred, y, mask_tensor=mask_tensor, mask_channel_outputs=self.mask_channel_outputs)
            elif self.loss_type == "absolute":
                return self.abs(y_pred, y, mask_tensor=mask_tensor, mask_channel_outputs=self.mask_channel_outputs)
        else:
            return self.hybrid(y_pred, y, mask_tensor=mask_tensor, mask_channel_outputs=self.mask_channel_outputs, loss_types = self.loss_type)

class H1Loss(object):
    """
    H1Loss provides the H1 Sobolev norm between
    two d-dimensional discretized functions.

    .. note :: 
        In function space, the Sobolev norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to H1Loss, we assume a regular 
        discretization and take ``1 / measure`` as the quadrature weights. 

    Parameters
    ----------
    d : int, optional
        dimension of input functions, by default 1
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        .. note::

        To perform quadrature, ``H1Loss`` scales ``measure`` by the size
        of each spatial dimension of ``x``, and multiplies them with 
        ||x-y||, such that the final norm is a scaled average over the spatial
        dimensions of ``x``. 

    reduction : str, optional
        whether to reduce across the batch and channel dimension
        by summing ('sum') or averaging ('mean')

        .. warning : 

            H1Loss always averages over the spatial dimensions. 
            `reduction` only applies to the batch and channel dimensions.
    fix_x_bnd : bool, optional
        whether to fix finite difference derivative
        computation on the x boundary, by default False
    fix_y_bnd : bool, optional
        whether to fix finite difference derivative
        computation on the y boundary, by default False
    fix_z_bnd : bool, optional
        whether to fix finite difference derivative
        computation on the z boundary, by default False
    """
    def __init__(self,
            d=1,
            measure=1.,
            reduction='sum',
            fix_x_bnd=False,
            fix_y_bnd=False,
            fix_z_bnd=False,
            data_processor=None,
            loss_type='relative',
            mask_channel_outputs=None,
            mask_channel=None):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd
        self.data_processor = data_processor
        # Ensure loss_type is either a string or a list of strings
        if not (isinstance(loss_type, str) or (isinstance(loss_type, list) and all(isinstance(l, str) for l in loss_type))):
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Expected a string or a list of strings.")

        self.loss_type = loss_type

        self.mask_channel_outputs = mask_channel_outputs
        self.mask_channel = mask_channel
        
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"H1_{self.d}DLoss"
     
    def compute_terms(self, x, y, quadrature):
        """compute_terms computes the necessary
        finite-difference derivative terms for computing
        the H1 norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : int or list
            quadrature weights

        """
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = central_diff_1d(x, quadrature[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, quadrature[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x
        
        elif self.d == 2:
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            x_x, x_y = central_diff_2d(x, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)
        
        else:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)
        
        return dict_x, dict_y

    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that 
        ``LpLoss`` computes the average over spatial dims. 

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature

    def reduce_channel(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x,dim=0)
        else:
            x = torch.mean(x,dim=0)
        
        return x
    
    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x)
        else:
            x = torch.mean(x)
        
        return x
        
    def abs(self, x, y, quadrature=None, mask_tensor=None, mask_channel_outputs=None):
        """absolute H1 norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature constant for reduction along each dim, by default None
        """
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
            
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        const = math.prod(quadrature)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += const*torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2

        if mask_tensor is not None and mask_channel_outputs is not None:
            # Ensure mask_channel_outputs is a tensor for proper indexing
            if isinstance(mask_channel_outputs, list):
                mask_channel_outputs = torch.tensor(mask_channel_outputs, device=x.device)

            mask_tensor = torch.flatten(mask_tensor, start_dim=-self.d)

            # Apply mask only on selected channels
            diff[:, mask_channel_outputs, ...] *= mask_tensor
        
        diff = diff**0.5

        diff_channel_wise = self.reduce_channel(diff)
        
        diff = self.reduce_all(diff).squeeze()
            
        return diff, diff_channel_wise
        
    def rel(self, x, y, quadrature=None, mask_tensor=None, mask_channel_outputs=None):
        """relative H1-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature constant for reduction along each dim, by default None
        """
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        if mask_tensor is not None and mask_channel_outputs is not None:
            # Ensure mask_channel_outputs is a tensor for proper indexing
            if isinstance(mask_channel_outputs, list):
                mask_channel_outputs = torch.tensor(mask_channel_outputs, device=x.device)

            mask_tensor = torch.flatten(mask_tensor, start_dim=-self.d)

            # Apply mask only on selected channels
            diff[:, mask_channel_outputs, ...] *= mask_tensor

        diff = (diff**0.5)/(ynorm**0.5)

        diff_channel_wise = torch.reduce_channel(diff)
        diff = self.reduce_all(diff).squeeze()
            
        return diff, diff_channel_wise

    def hybrid(self, x, y, quadrature=None, mask_tensor=None, mask_channel_outputs=None, loss_types=None):
        """relative H1-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature constant for reduction along each dim, by default None
        """
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        if mask_tensor is not None and mask_channel_outputs is not None:
            # Ensure mask_channel_outputs is a tensor for proper indexing
            if isinstance(mask_channel_outputs, list):
                mask_channel_outputs = torch.tensor(mask_channel_outputs, device=x.device)

            mask_tensor = torch.flatten(mask_tensor, start_dim=-self.d)

            # Apply mask only on selected channels
            diff[:, mask_channel_outputs, ...] *= mask_tensor

        #diff = (diff**0.5)/(ynorm**0.5)
        # Ensure loss_types has the correct length
        num_channels = diff_norm.shape[1]  # The second dimension is the channel dimension
        if len(loss_types) != num_channels:
            raise ValueError(f"Mismatch: loss_types has {len(loss_types)} elements, but expected {num_channels}.")
        
        diff = torch.zeros_like(diff_norm)
        # Iterate over channels and apply different loss calculations
        for channel in range(num_channels):
            if loss_types[channel] == "relative":
                diff[:, channel, ...] = (diff_norm[:, channel, ...]**0.5) / (y_norm[:, channel, ...]**0.5)
            else:
                diff[:, channel, ...] = (diff_norm[:, channel, ...]**0.5)  # Keep absolute error

        diff_channel_wise = torch.reduce_channel(diff)
        diff = self.reduce_all(diff).squeeze()
            
        return diff, diff_channel_wise

    def __call__(self, y_pred, y, quadrature=None, **kwargs):
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            normalization constant for reduction, by default None
        """
        input_x = kwargs['x'].clone()
        mask_tensor = None
        if self.data_processor is not None:
            if self.data_processor.in_normalizer is not None:
                input_x = self.data_processor.in_normalizer.inverse_transform(input_x)
                mask_tensor = input_x[:,self.mask_channel:self.mask_channel+1,:,:,:]
        if isinstance(self.loss_type, str):
            if self.loss_type == "relative":
                return self.rel(y_pred, y, quadrature=quadrature, mask_tensor=mask_tensor, mask_channel_outputs=self.mask_channel_outputs,)
            elif self.loss_type == "absolute":
                return self.abs(y_pred, y, quadrature=quadrature, mask_tensor=mask_tensor, mask_channel_outputs=self.mask_channel_outputs,)
        else:
            return self.hybrid(y_pred, y, mask_tensor=mask_tensor, mask_channel_outputs=self.mask_channel_outputs, loss_types = self.loss_type)

