import torch
import torch.nn as nn
import torch.nn.functional as func


class Total_loss(nn.Module):
    def __init__(self, lamb=0.5):
        super(Total_loss, self).__init__()
        self.lamb = lamb
        self.criterion_bcel = nn.BCELoss()
        self.criterion_cmfl = CMFLoss() 
    
    def forward(self, p, q, r, targets):
        """
        Args:
            p: probability of real in rgb branch
            q: probability of real in depth branch
            r: probability of real in joint branch
            targets: {0:fake, 1:real}
        """
        bce_loss_r = self.criterion_bcel(r, targets) # CE(rt) = BCE(r)
		# old CMFL
        cmf_loss_pq = self.criterion_cmfl(p,q,targets) 
		# new CMFL
		# cmf_loss_pq = self.criterion_cmfl(p,q,targets)+self.criterion_cmfl(p,q,targets) 
        loss = (1-self.lamb)*bce_loss_r + self.lamb*cmf_loss_pq
        return loss

class CMF_loss(nn.Module):
	"""
	Cross Modal Focal Loss
	Args:
		alpha: alpha balanced
		gamma: modulating factor
		multiplier: num of branches
	"""
	def __init__(self, alpha=0.75, gamma=3, multiplier=2):
		super(CMF_loss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.multiplier =multiplier
	
	def forward(self, p, q, targets):
		""""
        Args:
            p: probability of "real" in rgb branch
            q: probability of "real" in depth branch
            r: probability of "real" in joint branch
        """
		bce_loss_p = func.binary_cross_entropy(p, targets, reduce=False) # CE(pt) = BCE(p)
		bce_loss_q = func.binary_cross_entropy(q, targets, reduce=False)

		pt = torch.exp(-bce_loss_p)	# prob of the target class in rgb branch
		qt = torch.exp(-bce_loss_q)

		cmfl_pq = self.alpha * (1-self.w(pt, qt))**self.gamma * bce_loss_p
		cmfl_qp = self.alpha * (1-self.w(qt, pt))**self.gamma * bce_loss_q
		cmfl = 0.5*torch.mean(cmfl_pq) + 0.5*torch.mean(cmfl_qp) 
		return cmfl

	def w(self, pt, qt):
		eps = 1e-8
		w = ((qt + eps)*(self.multiplier*pt*qt))/(pt + qt + eps)
		return w

class CMFLoss(nn.Module):
	"""
	Cross Modal Focal Loss (old version)
	"""
	def __init__(self, alpha=1, gamma=2, binary=False, multiplier=2, sg=False):
		super(CMFLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.binary = binary
		self.multiplier =multiplier
		self.sg=sg

	def forward(self, inputs_a,inputs_b, targets):

		bce_loss_a = func.binary_cross_entropy(inputs_a, targets, reduce=False)	# CE(pt) = BCE(pt)
		bce_loss_b = func.binary_cross_entropy(inputs_b, targets, reduce=False)	# CE(qt) = BCE(qt)

		pt_a = torch.exp(-bce_loss_a)	# pt
		pt_b = torch.exp(-bce_loss_b)	# qt

		eps = 0.000000001	

		if self.sg:
			d_pt_a=pt_a.detach()
			d_pt_b=pt_b.detach()
			wt_a=((d_pt_b + eps)*(self.multiplier*pt_a*d_pt_b))/(pt_a + d_pt_b + eps)
			wt_b=((d_pt_a + eps)*(self.multiplier*d_pt_a*pt_b))/(d_pt_a + pt_b + eps)
		else:
			wt_a=((pt_b + eps)*(self.multiplier*pt_a*pt_b))/(pt_a + pt_b + eps)	# w(pt,qt)
			wt_b=((pt_a + eps)*(self.multiplier*pt_a*pt_b))/(pt_a + pt_b + eps)	# w(qt,pt)

		if self.binary:
			wt_a=wt_a * (1-targets)
			wt_b=wt_b * (1-targets)

		f_loss_a = self.alpha * (1-wt_a)**self.gamma * bce_loss_a	# CMFL(pt,qt)
		f_loss_b = self.alpha * (1-wt_b)**self.gamma * bce_loss_b	# CMFL(qt,pt)

		loss= 0.5*torch.mean(f_loss_a) + 0.5*torch.mean(f_loss_b) 
		
		return loss