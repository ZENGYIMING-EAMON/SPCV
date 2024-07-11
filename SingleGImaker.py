import os;import pymeshlab; import trimesh; 
from tqdm import tqdm;  import pytorch3d.ops; 
import itertools; from pytorch3d.loss import chamfer_distance; import json; 
from sklearn.neighbors import KDTree; from PIL import Image
from CD.chamferdist.chamfer import knn_points as knn_gpu
from KNN_CPU.lib.python import nearest_neighbors as knn_cpu
import open3d as o3d
import time

import torch
import torch.nn as nn; import torch.autograd as ag;
import torch.nn.functional as F;
import numpy as np
import random
import os

def set_seed(seed):
    # Fix the seed for Python's built-in random module
    random.seed(seed)

    # Fix the seed for NumPy
    np.random.seed(seed)

    # Fix the seed for PyTorch
    torch.manual_seed(seed)

    # If using CUDA (GPU), also fix the seed for the GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure reproducibility for operations on CUDA (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_seed_from_env(env_var='SEED'):
    seed = os.getenv(env_var)
    if seed is not None:
        return int(seed)
    raise ValueError(f"Environment variable {env_var} is not set")

# Load the seed from the environment variable
seed = load_seed_from_env()

# Set the seed
set_seed(seed)

# Save the seed to a file if needed
def save_seed(seed, file_path='seed.txt'):
    with open(file_path, 'w') as f:
        f.write(str(seed))

save_seed(seed)

# Example operations to demonstrate reproducibility
random_number = random.random()
numpy_array = np.random.rand(3)
torch_tensor = torch.rand(3)

print("Random number (random):", random_number)
print("Numpy array (numpy):", numpy_array)
print("Torch tensor (torch):", torch_tensor)
print("Used seed:", seed)

'''tool box'''
def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def scale_mesh(mesh_path,output_path):
    input=trimesh.load(mesh_path)
    mesh=as_mesh(input)
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)
    mesh.export(output_path)

def pds(mesh_path, num_pts, output_pc_path):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # Start the timer
    start_time = time.time()
    
    # Perform Poisson disk sampling
    pcd = mesh.sample_points_poisson_disk(num_pts)
    
    # Stop the timer and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Time taken for Poisson disk sampling: {elapsed_time:.2f} seconds")
    
    # Convert to numpy array and save as .xyz
    np.savetxt(output_pc_path, np.asarray(pcd.points))

class MLP_Decoder(nn.Module):
    def __init__(self, feat_dims=512):
        super(MLP_Decoder, self).__init__()
        
        self.folding1 = nn.Sequential(
                nn.Conv1d(2, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, feat_dims, 1),
            )

        self.folding2 = nn.Sequential(
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, 3, 1),
        )
    def forward(self, points):
        # points (B,N,2)
        cat1 = points.transpose(1,2)   
        folding_result1 = self.folding1(cat1)           # (batch_size, 3, num_points)
        cat2 = folding_result1 
        folding_result2 = self.folding2(cat2)           # (batch_size, 3, num_points)
        return folding_result2.transpose(1, 2)          # (batch_size, num_points ,3)

def df(x, wrt):
    B, M = x.shape
    return ag.grad(x.flatten(), wrt,
                    grad_outputs=torch.ones(B * M , dtype=torch.float32).
                    to(x.device), create_graph=True)[0]
    
def gradient(xyz,uv):
    x,y,z=xyz[...,0],xyz[...,1],xyz[...,2]
    dx=df(x,uv).unsqueeze(2)     #(B,N,1,2)
    dy=df(y,uv).unsqueeze(2)
    dz=df(z,uv).unsqueeze(2)

    dxyz=torch.cat((dx,dy,dz),dim=2)    #(B,N,3,2)
    return dxyz

class geo_loss_v3(nn.Module):
    # assign weight for each query
    def __init__(self,up_ratio=10,K=5,std_factor=10,weighted_query=True):
        super(geo_loss_v3,self).__init__()
        self.K=K
        self.up_ratio=up_ratio
        self.std_factor=std_factor
        self.weighted_query=weighted_query

    def cal_udf_weights(self,x,query):
        #x: (B,N,3)
        dists,idx,knn_pc=pytorch3d.ops.knn_points(query,x,K=self.K,return_nn=True,return_sorted=True)   #(B,N,K) (B,N,K) (B,N,K,3)
        dir=query.unsqueeze(2)-knn_pc   #(B,N,K,3)
        weights=torch.softmax(-dists,dim=2)   #(B,N,K) 
        udf=torch.sum((dists+1e-10).sqrt()*weights,dim=2)  #(B,N)
        udf_grad=torch.sum(dir*weights.unsqueeze(-1),dim=2) #(B,N,3)
        return udf,udf_grad,weights

    def cal_udf(self,x,weights,query):
        dists,idx,knn_pc=pytorch3d.ops.knn_points(query,x,K=self.K,return_nn=True,return_sorted=True)   #(B,N,K) (B,N,K) (B,N,K,3)
        dir=query.unsqueeze(2)-knn_pc   #(B,N,K,3)
        udf=torch.sum((dists+1e-10).sqrt()*weights,dim=2)  #(B,N)
        udf_grad=torch.sum(dir*weights.unsqueeze(-1),dim=2) #(B,N,3)
        return udf,udf_grad

    def forward(self,src,tgt):
        #src: target (B,N,3)
        #tgt: source (B,N,3)

        with torch.no_grad():
            tgt_self_dists,_,_=pytorch3d.ops.knn_points(tgt,tgt,return_nn=True,K=2,return_sorted=True)
            tgt_self_dists=tgt_self_dists[:,:,1:]   #(B,N,1)
            tgt_self_dists=torch.sqrt(tgt_self_dists+1e-10)
            std=tgt_self_dists*self.std_factor
            noise_offset=torch.randn(tgt.size(0),tgt.size(1),self.up_ratio,3).to(tgt).float() * std.unsqueeze(3)
            query=tgt.unsqueeze(2)+noise_offset
            query=query.reshape(tgt.size(0),-1,3).detach()
        query=torch.cat((query,src),dim=1)
        udf_tgt,udf_grad_tgt,weights=self.cal_udf_weights(tgt,query)
        udf_src,udf_grad_src=self.cal_udf(src,weights,query)
        udf_error=torch.abs(udf_tgt-udf_src)    #(B,M)
        udf_grad_error=torch.sum(torch.abs(udf_grad_src-udf_grad_tgt),axis=-1)  #(B,M)
        if self.weighted_query:
            with torch.no_grad():
                query_weights=torch.exp(-udf_error.detach()*3)*torch.exp(-udf_grad_error.detach()*3)
            return torch.sum((udf_error+udf_grad_error)*query_weights.detach())/query.size(0)/query.size(1)
        else:
            query_weights=1
            return torch.sum((udf_error+udf_grad_error)*query_weights)/query.size(0)/query.size(1)

class cd_func(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,src,tgt): 
        self.wgeo = 1.
        return self.wgeo*chamfer_distance(src,tgt,batch_reduction='sum',point_reduction='mean')[0]
cd = cd_func()

class geo_(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo = geo_loss_v3(10,5,3,False)

    def forward(self,src,tgt):
        self.wgeo = 1.
        return self.wgeo*self.geo(src,tgt)
geo = geo_()

def build_unit_grid(H, W):
    # build a unit square grid
    h_data = np.linspace(0, 1, H, dtype=np.float32)
    w_data = np.linspace(0, 1, W, dtype=np.float32)
    grid_h_by_w = np.array(list(itertools.product(h_data, w_data))) # (H*W, 2)
    return grid_h_by_w

def cal_gi_smooth_h_kdtree(H,W,rec_pts:np.array,J :int = 100,J_bar:int = 200):
    # rec_pts (N,3)
    rec_tree=KDTree(rec_pts)

    ebd=build_unit_grid(H, W)  #[N, 2]

    ebd_tree=KDTree(ebd)

    _,knn_index_2d=ebd_tree.query(ebd,k=J+1)
    knn_index_2d=knn_index_2d[:,1:]     #[N, J]

    _,knn_index_3d=rec_tree.query(rec_pts,k=J_bar+1)
    knn_index_3d=knn_index_3d[:,1:]      #[N, J_bar]

    percent_perpoint=np.sum(np.sum(np.equal(knn_index_2d[:,:,None],knn_index_3d[:,None,:]) ,axis=-1),axis=-1)/J    #(N,J,J_bar)

    percent=np.mean(percent_perpoint)

    return percent

def cal_gi(H,W,metric_dict,points,normals=None):
    J=8; J_bar_set=[8,16,32,64,128]
    for J_bar in J_bar_set:
        percent = cal_gi_smooth_h_kdtree(H,W,points,J,J_bar)
        metric_dict[f'jbar_{J_bar}'] = '{:.4f}'.format(percent*100)+"%"
        
def normalize_to_01(nc_array):
    AA = nc_array.copy()
    AA -= np.min(AA)
    AA /= np.max(AA)
    return AA

def write_gi(save_path,points,H,W,normals=None):
    N,C = points.shape
    points = normalize_to_01(points) 
    gi = np.expand_dims(points, axis=0) #1NC
    gi = np.transpose(gi, (0, 2, 1))     #1CN
    gi = gi.reshape(1, C, H,W)
    _, _, H, W = gi.shape
    y_pos = np.arange(0, H); z_pos = np.arange(0, W)
    img_ = np.zeros((H, W, 3), dtype=np.uint8) 
    for _y in y_pos.tolist():
        for _z in z_pos.tolist():
            c = (gi[0][:, _y, _z]*255.0)
            img_[_y, _z] = c #255
    data = img_
    img = Image.fromarray(data, 'RGB')
    img.save(save_path) 
            
class smooth_constraints(nn.Module):
    def __init__(self, ks=3):
        super(smooth_constraints,self).__init__()
        kernel = (ks, ks)
        self.uf = torch.nn.Unfold(kernel)
        self.winH, self.winW = kernel[0], kernel[1]
    
    def forward(self, dIMG):#B3HW
        pp = self.uf(dIMG)
        bs, C, H, W = dIMG.shape
        pp_find_mid_shape = pp.reshape(bs,C, self.winH, self.winW, -1) #bsize channel windowH windowW how_many_windows
        pp_middle = pp_find_mid_shape[:,:,1,1,:] #torch.Size([B, C, howmanywindows]) 

        pp_cal_mean_shape = pp.reshape(bs,C, self.winH*self.winW, -1)
        pp_mean = torch.mean(pp_cal_mean_shape, dim=2) #torch.Size([B, C, howmanywindows]) 

        loss_smth = torch.mean(torch.abs(pp_middle-pp_mean)) 
        return loss_smth
            
class geo_smth_nrm_flexcut(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo = geo_loss_v3(10,5,3,False)
        self.smth = smooth_constraints(ks=3)
        self.fsmth = lambda x, a: (1/a)**2 * (x-a)**2

    def forward(self,src,tgt,grid, ep, maxep, H, W):
        if ep < 1000:
            rec_pc = src
            grad=gradient(rec_pc,grid)
            grad_u,grad_v=grad[...,0],grad[...,1]
            normal=torch.cross(grad_u,grad_v)
            normal=F.normalize(normal,dim=-1) 
        
        self.wsmth = self.fsmth(ep, maxep)
        self.wgeo = 1.
        print('ep= {}, weight geo = {} ; weight smth = {}'.format(ep, self.wgeo, self.wsmth))
        B, N, C = src.shape
        src_img = src.permute(0,2,1).reshape(B,C,H,W).contiguous()
        if ep < 1000:
            src_nrm = normal.permute(0,2,1).reshape(B,C,H,W).contiguous()
            return self.wgeo*self.geo(src,tgt) + self.wsmth*self.smth(src_img) + self.wsmth*self.smth(src_nrm)
        elif ep >=1000:
            return self.wgeo*self.geo(src,tgt) + self.wsmth*self.smth(src_img)

def build_grid(H, W,delta=0.3):
    meshgrid = [[-delta, delta, H], [-delta, delta, W]]
    
    x = np.linspace(*meshgrid[0])
    y = np.linspace(*meshgrid[1])
    points = np.array(list(itertools.product(x, y))).astype(np.float32)    #(B,res*res,2)
    
    return points
             
class fit_optim(object):
    def __init__(self,ep, H, W):
        self.loss_func = geo_smth_nrm_flexcut()
        self.decoder=MLP_Decoder().cuda()
        self.grid_=build_grid(H,W)    #(2000,2)
        self.ep=ep
        self.H = H; self.W = W

    def __call__(self, gt_pc):
        N,C = gt_pc.shape
        self.grid=torch.from_numpy(self.grid_).cuda().float().unsqueeze(0)
        self.grid.requires_grad=True 
        optimizer=torch.optim.Adam(self.decoder.parameters(),lr=1e-3,weight_decay=0.00001)
        for i in tqdm(range(1,self.ep+1)):
            self.decoder.train()
            rec_pc=self.decoder(self.grid)
            loss=self.loss_func(rec_pc,torch.from_numpy(gt_pc).unsqueeze(0).float().cuda(),self.grid, i-1, self.ep, self.H, self.W) #*1000
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return rec_pc.squeeze(0).detach().cpu().numpy()

class fl_optim_geo(object):
    def __init__(self, Wsmth=20):
        self.Wsmth = Wsmth

    def __call__(self, src, tgt, ep=1000):
        src = torch.from_numpy(src).unsqueeze(0).cuda().float()
        tgt = torch.from_numpy(tgt).unsqueeze(0).cuda().float()
        '''BN3 BN3'''
    
        loss_func = geo_loss_v3(10,5,3,False) 
        flow=torch.zeros_like(src).float().cuda()*0
        flow.requires_grad=True
        optimizer=torch.optim.Adam([flow],lr=1e-2)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,ep,1e-3)
        
        for i in tqdm(range(1,ep+1)):
            pred_tgt=src+flow
            dists,knn_index,knn_src=pytorch3d.ops.knn_points(src,src,K=30,return_nn=True,return_sorted=True)
            knn_flow=pytorch3d.ops.knn_gather(flow,knn_index)
            flow_diff=flow.unsqueeze(2)-knn_flow[:,:,1:,:]
            weights=1/(1e-8+dists[:,:,1:])  #(B,N,K)
            weights=weights/torch.sum(weights,dim=2,keepdim=True)
            smooth_error=torch.sum(weights*torch.square(flow_diff).sum(dim=3),dim=2)
                
            loss=1*loss_func(pred_tgt,tgt)+self.Wsmth*torch.mean(smooth_error) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return pred_tgt.squeeze(0).detach().cpu().numpy(), flow.squeeze(0).detach().cpu().numpy()

class fl_optim_cd(object):
    def __init__(self, Wsmth=20):
        self.Wsmth = Wsmth
        self.fsmth = lambda x, a: (1/a)**2 * (x-a)**2
    
    def __call__(self, src, tgt, ep=1000):
        src = torch.from_numpy(src).unsqueeze(0).cuda().float()
        tgt = torch.from_numpy(tgt).unsqueeze(0).cuda().float()
        loss_func = cd_func() #几乎不会改变结果 #只会降低一点点cd
        
        flow=torch.zeros_like(src).float().cuda()*0
        flow.requires_grad=True
        optimizer=torch.optim.Adam([flow],lr=1e-2)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,ep,1e-3)
        
        for i in tqdm(range(1,ep+1)):
            pred_tgt=src+flow
            dists,knn_index,knn_src=pytorch3d.ops.knn_points(src,src,K=30,return_nn=True,return_sorted=True)
            knn_flow=pytorch3d.ops.knn_gather(flow,knn_index)
            flow_diff=flow.unsqueeze(2)-knn_flow[:,:,1:,:]
            weights=1/(1e-8+dists[:,:,1:])  #(B,N,K)
            weights=weights/torch.sum(weights,dim=2,keepdim=True)
            smooth_error=torch.sum(weights*torch.square(flow_diff).sum(dim=3),dim=2)
            if i > int(0.1*ep): 
                self.Wsmth = self.fsmth(i, ep)
                
            loss=1*loss_func(pred_tgt,tgt)+self.Wsmth*torch.mean(smooth_error) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return pred_tgt.squeeze(0).detach().cpu().numpy(), flow.squeeze(0).detach().cpu().numpy()
        
class shape_self_refine(object):
    def __init__(self, Wsmth=20):
        self.op1 = fl_optim_geo(Wsmth=Wsmth)
        self.op2 = fl_optim_cd(Wsmth=Wsmth)

    def __call__(self, src, tgt, ep=1000):
        rec_pc,_ = self.op1(src, tgt, ep)
        rec_pc,_ = self.op2(rec_pc, tgt, ep)
        return rec_pc

def index_points(pc, idx):
    # pc: [B, N, C]
    # 1) idx: [B, S] -> pc_selected: [B, S, C]
    # 2) idx: [B, S, K] -> pc_selected: [B, S, K, C]
    device = pc.device
    B = pc.shape[0]
    view_shape = list(idx.shape) 
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_shape).repeat(repeat_shape)
    pc_selected = pc[batch_indices, idx, :]
    return pc_selected

def knn_on_gpu(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == 'cuda'
    assert query_pts.device.type == 'cuda'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_gpu(p1=query_pts, p2=source_pts, K=k, return_nn=False, return_sorted=True)[1]
    return knn_idx

def knn_on_cpu(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == 'cpu'
    assert query_pts.device.type == 'cpu'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_cpu.knn_batch(source_pts, query_pts, k, omp=True)
    return knn_idx

def knn_search(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == query_pts.device.type
    device_type = source_pts.device.type
    assert device_type in ['cpu', 'cuda']
    if device_type == 'cuda':
        knn_idx = knn_on_gpu(source_pts, query_pts, k)
    if device_type == 'cpu':
        knn_idx = knn_on_cpu(source_pts, query_pts, k)
    return knn_idx

def rearr_gt(struc_pc, gt_pc):
    gt_pc = torch.from_numpy(gt_pc).unsqueeze(0)
    struc_pc = torch.from_numpy(struc_pc).unsqueeze(0)
    para_a = index_points(gt_pc, knn_search(gt_pc, struc_pc, 1).squeeze(-1)) # [B, M, 3]
    return para_a.squeeze(0).numpy()

def main_v1_1(mesh_path, ep_1stfit=20000, ep_rf=1000, num_pts=10000, H=100, W=100):
    num_pts = num_pts
    H = H
    W = W           
    assert H*W == num_pts
    ep_1stfit = ep_1stfit 
    ep_rf = ep_rf    
    np_random_permute_seed = 123
    np.random.seed(np_random_permute_seed) 
    arr = np.array([1, 2, 3, 4, 5])
    shuffled_arr = np.random.permutation(arr) 

    ext = mesh_path.split('.')[-1] 
    sample_pc_path = mesh_path
    struc_pc_path = mesh_path.replace(f'.{ext}',f'.pds_{num_pts}_structure.xyz')
    struc_gi_path = mesh_path.replace(f'.{ext}',f'.pds_{num_pts}_structure.png')
    refine_struc_pc_path = mesh_path.replace(f'.{ext}',f'.pds_{num_pts}_structure_refine.xyz')
    refine_struc_gi_path = mesh_path.replace(f'.{ext}',f'.pds_{num_pts}_structure_refine.png')
    rearr_refine_struc_pc_path = mesh_path.replace(f'.{ext}',f'.pds_{num_pts}_structure_refine_rearr.xyz')
    rearr_refine_struc_gi_path = mesh_path.replace(f'.{ext}',f'.pds_{num_pts}_structure_refine_rearr.png')
    save_metric_path = mesh_path.replace(f'.{ext}',f'.pds_{num_pts}_metric.json')

    flag = {
        'skip_scale_mesh': True,
        'skip_sample_pc': True,
        'skip_generate_struc_pc': False,
        'skip_generate_refine_struc_pc': False,
        'skip_rearrage_gt_to_refine_struc_pc': False,
    }
    if os.path.exists(sample_pc_path):
        flag['skip_sample_pc'] = True
    if os.path.exists(struc_pc_path):
        flag['skip_generate_struc_pc'] = True
    if os.path.exists(refine_struc_pc_path):
        flag['skip_generate_refine_struc_pc'] = True
    if os.path.exists(rearr_refine_struc_pc_path):
        flag['skip_rearrage_gt_to_refine_struc_pc'] = True

    metric = {}
    metric['gi_bf_refine'] = {}
    metric['gi_af_refine'] = {}
    gt_pc = np.loadtxt(sample_pc_path)
    op_1st = fit_optim(ep_1stfit, H, W)
    op_ssrf = shape_self_refine()

    if flag['skip_generate_struc_pc'] == False:
        rec_pc = op_1st(gt_pc)
        np.savetxt(struc_pc_path, rec_pc)
        rec_pc = np.loadtxt(struc_pc_path)
        write_gi(struc_gi_path, rec_pc, H=H, W=W, normals=None)
        cal_gi(H=H,W=W,metric_dict=metric['gi_bf_refine'],points=rec_pc,normals=None)
        metric['geo_bf_refine'] = geo(torch.from_numpy(gt_pc).unsqueeze(0).cuda().float(), \
                            torch.from_numpy(rec_pc).unsqueeze(0).cuda().float()).item()
        metric['cd_bf_refine'] = cd(torch.from_numpy(gt_pc).unsqueeze(0).cuda().float(), \
                            torch.from_numpy(rec_pc).unsqueeze(0).cuda().float()).item()

    if flag['skip_generate_refine_struc_pc'] == False:
        rec_pc = op_ssrf(rec_pc, gt_pc, ep_rf)
        np.savetxt(refine_struc_pc_path, rec_pc)
        rec_pc = np.loadtxt(refine_struc_pc_path)
        write_gi(refine_struc_gi_path, rec_pc, H=H, W=W, normals=None)
        cal_gi(H=H,W=W,metric_dict=metric['gi_af_refine'],points=rec_pc,normals=None)
        metric['geo_af_refine'] = geo(torch.from_numpy(gt_pc).unsqueeze(0).cuda().float(), \
                            torch.from_numpy(rec_pc).unsqueeze(0).cuda().float()).item()
        metric['cd_af_refine'] = cd(torch.from_numpy(gt_pc).unsqueeze(0).cuda().float(), \
                            torch.from_numpy(rec_pc).unsqueeze(0).cuda().float()).item()
        print(metric)
        with open(save_metric_path, "w") as f:
            json.dump(metric, f, indent=4)
            
    if flag['skip_rearrage_gt_to_refine_struc_pc'] == False:
        rec_pc_old = np.loadtxt(refine_struc_pc_path)
        rec_pc = rearr_gt(rec_pc_old, gt_pc)
        l1_loss = np.mean(np.abs(rec_pc - rec_pc_old))
        np.savetxt(rearr_refine_struc_pc_path, rec_pc)
        
        # Open the JSON file and read its contents
        with open(save_metric_path, 'r') as f:
            metric = json.load(f)
        metric['rec_vs_rec_old_l1_loss'] = l1_loss    
        metric['gi_af_refine_rearrageGT'] = {}
        write_gi(rearr_refine_struc_gi_path, rec_pc, H=H, W=W, normals=None)
        cal_gi(H=H,W=W,metric_dict=metric['gi_af_refine_rearrageGT'],points=rec_pc,normals=None)
        metric['geo_af_refine_rearrageGT'] = geo(torch.from_numpy(gt_pc).unsqueeze(0).cuda().float(), \
                            torch.from_numpy(rec_pc).unsqueeze(0).cuda().float()).item()
        metric['cd_af_refine_rearrageGT'] = cd(torch.from_numpy(gt_pc).unsqueeze(0).cuda().float(), \
                            torch.from_numpy(rec_pc).unsqueeze(0).cuda().float()).item()
        print(metric)
        with open(save_metric_path, "w") as f:
            json.dump(metric, f, indent=4)

def create_json_from_dict(dict, dir_):
    # Open a file in write mode and write the JSON object to it
    with open(dir_, 'w') as f:
        json.dump(dict, f, indent=4)
    # Close the file
    f.close()

def load_json(dir_):
    # Open the JSON file and load the data into a Python object
    with open(dir_, 'r') as f:
        dict = json.load(f)
    # Close the file
    f.close()
    return dict

if __name__ == '__main__':

    ml = [
        "./swing_pick_objs_sample1w_fpsUnify/0025.xyz",
    ]
    
    for p in ml:
        main_v1_1(mesh_path=p,  ep_1stfit=2, ep_rf=20, num_pts=10000, H=100, W=100)
        



