import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as func


class Total_loss(nn.Module):
	def __init__(self, device, lamb=0.5, alpha=1, gamma=3):
		super(Total_loss, self).__init__()
		self.lamb = lamb
		self.bcel = nn.BCELoss()
		self.cmfl = CMFLoss(alpha, gamma)

	def forward(self, p, q, r, targets):
		"""
		Args:
		p: prob of live in rgb branch	[B,1]
		q: prob of live in depth branch	[B,1]
		r: prob of live in joint branch	[B,1]
		targets: {0:fake, 1:live}
		"""
		bcel_r = self.bcel(r, targets) 		# CE(rt) = BCE(r)
		# print('[Loss-BCE]\tbcel_r: {}'.format(bcel_r.item()))
		cmfl_pq = self.cmfl(p,q,targets)	# CMFL(pt,qt)+CMFL(qt,pt)
		# print('[Loss-CMFL]\tcmfl_pq: {}'.format(cmfl_pq.item()))
		error = (1-self.lamb)*bcel_r + self.lamb*cmfl_pq
		return error

class CMFLoss(nn.Module):
	"""
	Cross Modal Focal Loss
	"""
	def __init__(self, alpha, gamma):
		"""
		Args:
			alpha: alpha balanced
			gamma: tunnable focusing parameter. modulating factor is (1-pt)**gamma = (1-w(pt,qt)**gamma)
			multiplier: num of branches
		"""
		super(CMFLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
	
	def forward(self, p, q, targets):
		""""
        Args:
            p: prob of live in rgb branch. 		[B,1]
            q: prob of live in depth branch. 	[B,1]
            r: prob of live in joint branch. 	[B,1]
        """
		bce_loss_p = func.binary_cross_entropy(p, targets, reduce=False) # CE(pt) = BCE(p)
		bce_loss_q = func.binary_cross_entropy(q, targets, reduce=False)

		pt = torch.exp(-bce_loss_p)	# prob of the target class in rgb branch
		qt = torch.exp(-bce_loss_q)

		cmfl_pq = self.alpha * (1-self.w(pt, qt))**self.gamma * bce_loss_p	# CMFL(pt,qt)
		cmfl_qp = self.alpha * (1-self.w(qt, pt))**self.gamma * bce_loss_q
		cmfl = 0.5*torch.mean(cmfl_pq) + 0.5*torch.mean(cmfl_qp) 
		return cmfl

	def w(self, pt, qt):
		"""
		Depends on the probabilities given by the channels from two individual branches
		"""
		eps = 1e-8
		w = ((qt + eps)*(2*pt*qt))/(pt + qt + eps)
		return w
