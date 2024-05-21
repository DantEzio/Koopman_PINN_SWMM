import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class NN(nn.Module):
    def __init__(self, n_input, n_output, n_h, Num_Layer):
        super(NN, self).__init__()
        self.Num_Layer = Num_Layer
        self.hencoding, self.hdecoding = [], []
        self.inlayer = nn.Linear(2 * n_input, n_h)
        for _ in range(self.Num_Layer):
            #fc = nn.Linear(n_h,n_h)          
            #setattr(self,'fc%i'%i,fc)
            self.hencoding.append(nn.Linear(n_h, n_h))
        self.K = nn.Linear(n_h, n_h)
        for _ in range(self.Num_Layer):
            #fc = nn.Linear(n_h,n_h)          
            #setattr(self,'fc%i'%i,fc)
            self.hdecoding.append(nn.Linear(n_h, n_h))
        self.outlayer = nn.Linear(n_h, n_output)

    def forward(self, x):
        x = torch.cat((torch.cos(x).view(-1,x.shape[1]),torch.sin(x).view(-1,x.shape[1])),axis=1)
        x = self.inlayer(x)
        for i in range(self.Num_Layer):                      
            x = F.tanh(self.hencoding[i](x))
            #x = self.b[i](x)
        x = self.K(x)
        for i in range(self.Num_Layer):                      
            x = F.tanh(self.hdecoding[i](x))
            #x = self.b[i](x)
        out = self.outlayer(x)
        return out
    
class meta_surrogate_a(nn.Module):
    def __init__(self,params):
        super(meta_surrogate_a, self).__init__()
        # 根据SWMM的dynwave计算过程设计计算图
        self.a_Layer = NN(params['alayer'][0],params['alayer'][1],params['alayer'][2],params['alayer'][3])#7, 1, 20, 5)
        self.Wslot_Layer = NN(params['wlayer'][0],params['wlayer'][1],params['wlayer'][2],params['wlayer'][3])#2, 1, 20, 5)
    
    def forward(self, h1, z1, h2, z2, aold, qlast, dt, yFull, aFull):
        # qlast and qold, how to deal with this?
        y1 = h1.view(-1,1) - z1.view(-1,1)
        y2 = h2.view(-1,1) - z2.view(-1,1)
        ymid = (y1+y2)/2
        Wslot2 = self.Wslot_Layer(torch.cat([y2.float().view(-1,1),
                                             yFull.float().view(-1,1)],axis=1))
        
        amid_tem = self.a_Layer(torch.cat([ymid.float().view(-1,1),
                                       Wslot2.float().view(-1,1),
                                       yFull.float().view(-1,1),
                                       aFull.float().view(-1,1),
                                       aold.float().view(-1,1),
                                       qlast.float().view(-1,1),
                                       dt.float().view(-1,1)],axis=1))
        
        amid = torch.clip(amid_tem, torch.tensor(np.zeros(aFull.shape)).float().view(-1,1), aFull.float().view(-1,1))
        return amid
    
    
    
class meta_surrogate_q(nn.Module):

    def __init__(self,param):
        super(meta_surrogate_q, self).__init__()
        # 根据SWMM的dynwave计算过程设计计算图
        self.Wslot_Layer = NN(param['wlayer'][0],param['wlayer'][1],param['wlayer'][2],param['wlayer'][3])
        self.r_Layer = NN(param['rlayer'][0],param['rlayer'][1],param['rlayer'][2],param['rlayer'][3])
        self.a_Layer = NN(param['alayer'][0],param['alayer'][1],param['alayer'][2],param['alayer'][3])
        self.v_Layer = NN(param['vlayer'][0],param['vlayer'][1],param['vlayer'][2],param['vlayer'][3])

        self.dq1_Layer = NN(param['dq1layer'][0],param['dq1layer'][1],param['dq1layer'][2],param['dq1layer'][3])
        self.dq2_Layer = NN(param['dq2layer'][0],param['dq2layer'][1],param['dq2layer'][2],param['dq2layer'][3])
        self.dq3_Layer = NN(param['dq3layer'][0],param['dq3layer'][1],param['dq3layer'][2],param['dq3layer'][3])
        self.dq4_Layer = NN(param['dq4layer'][0],param['dq4layer'][1],param['dq4layer'][2],param['dq4layer'][3])
        self.dq5_Layer = NN(param['dq5layer'][0],param['dq5layer'][1],param['dq5layer'][2],param['dq5layer'][3])
        self.dq6_Layer = NN(param['dq6layer'][0],param['dq6layer'][1],param['dq6layer'][2],param['dq6layer'][3])
        
        self.q_Layer = NN(param['qlayer'][0],param['qlayer'][1],param['qlayer'][2],param['qlayer'][3])
    
    def forward(self, h1, z1, h2, z2, length, sigma, aold, amid, qlast, dt, yFull, aFull, rFull):
        # qlast and qold, how to deal with this?
        y1 = h1.view(-1,1) - z1.view(-1,1)
        y2 = h2.view(-1,1) - z2.view(-1,1)
        ymid = (y1+y2)/2

        Wslot1 = self.Wslot_Layer(torch.cat([y1.float().view(-1,1),yFull.float().view(-1,1)],axis=1))
        Wslot2 = self.Wslot_Layer(torch.cat([y2.float().view(-1,1),yFull.float().view(-1,1)],axis=1))

        a1 = self.a_Layer(torch.cat([y1.float().view(-1,1),
                                     Wslot1.float().view(-1,1),
                                     yFull.float().view(-1,1),
                                     aFull.float().view(-1,1)],axis=1))
        a2 = self.a_Layer(torch.cat([y2.float().view(-1,1),
                                     Wslot2.float().view(-1,1),
                                     yFull.float().view(-1,1),
                                     aFull.float().view(-1,1)],axis=1))

        r1 = self.r_Layer(torch.cat([y1.float().view(-1,1),
                                     yFull.float().view(-1,1),
                                     rFull.float().view(-1,1)],axis=1))
        rmid = self.r_Layer(torch.cat([ymid.float().view(-1,1),
                                       yFull.float().view(-1,1),
                                       rFull.float().view(-1,1)],axis=1))
        
        rwtd = r1 + (rmid - r1)
        awtd = a1 + (amid.float().view(-1,1) - a1)
        
        v = self.v_Layer(torch.cat([qlast.float().view(-1,1), 
                                    amid.float().view(-1,1)],axis=1))#qlast.view(-1,1)/amid.view(-1,1)
        
        dq1 = dt.view(-1,1) * self.dq1_Layer(torch.cat([v.float().view(-1,1),
                                                      rwtd.float().view(-1,1)],axis=1))
        dq2 = dt.view(-1,1) * 9.8 * self.dq2_Layer(torch.cat([h1.float().view(-1,1),
                                                            h2.float().view(-1,1),
                                                            awtd.float().view(-1,1),
                                                            length.float().view(-1,1)],axis=1))
        dq3 = 2 * sigma.view(-1,1) * self.dq3_Layer(torch.cat([v.float().view(-1,1),
                                                             sigma.float().view(-1,1),
                                                             amid.float().view(-1,1),
                                                             aold.float().view(-1,1)],axis=1))
        dq4 = dt.view(-1,1) * sigma.view(-1,1) * self.dq4_Layer(torch.cat([v.float().view(-1,1),
                                                              a1.float().view(-1,1),
                                                              a2.float().view(-1,1),
                                                              length.float().view(-1,1)],axis=1))
        dq5 = 2 * length.view(-1,1) * dt.view(-1,1) * self.dq5_Layer(torch.cat([a1.float().view(-1,1),
                                                               qlast.float().view(-1,1),
                                                               amid.float().view(-1,1),
                                                               a2.float().view(-1,1)],axis=1))
        dq6 = (dt.view(-1,1) * 2.5 * v.view(-1,1) / length.view(-1,1)) * self.dq6_Layer(torch.cat([qlast.float().view(-1,1),dt.float().view(-1,1)],axis=1))
        q = self.q_Layer(torch.cat([qlast.float().view(-1,1),
                                    dq1.float().view(-1,1),
                                    dq2.float().view(-1,1),
                                    dq3.float().view(-1,1),
                                    dq4.float().view(-1,1),
                                    dq5.float().view(-1,1),
                                    dq6.float().view(-1,1)],axis=1))#(qlast.view(-1,1)-dq2+dq3+dq4+dq5+dq6)/denom
        return q



class iter_surrogate_aq(nn.Module):
    def __init__(self,parama,paramq):
        super(iter_surrogate_aq, self).__init__()
        # 在上述基础上构建迭代过程
        self.anet = meta_surrogate_a(parama)
        self.qnet = meta_surrogate_q(paramq)

    def forward(self, h1, z1, h2, z2, length, sigma, aold, qlast, dt, yFull, aFull, rFull):

        aresults,qresults = [aold],[qlast]
        for _ in range(3):
            a = self.anet(h1.float().view(-1,1),z1.float().view(-1,1),
                          h2.float().view(-1,1),z2.float().view(-1,1),
                          aresults[-1].float().view(-1,1),qresults[-1].float().view(-1,1),
                          dt.float().view(-1,1),yFull.float().view(-1,1),
                          aFull.float().view(-1,1))
        
            q = self.qnet(h1.float().view(-1,1),z1.float().view(-1,1),
                          h2.float().view(-1,1),z2.float().view(-1,1),
                          length.float().view(-1,1),sigma.float().view(-1,1),
                          aresults[-1].float().view(-1,1),a.float().view(-1,1),
                          qresults[-1].float().view(-1,1),dt.float().view(-1,1),
                          yFull.float().view(-1,1),aFull.float().view(-1,1),rFull.float().view(-1,1))
            
            aresults.append(a)
            qresults.append(q)

        return aresults,qresults


class meta_surrogate_h(nn.Module):
    def __init__(self,params):
        super(meta_surrogate_h, self).__init__()
        # 根据SWMM的dynwave计算过程设计计算图
        self.h_Layer = NN(params['hlayer'][0],params['hlayer'][1],params['hlayer'][2],params['hlayer'][3])#7, 1, 20, 5)
    
    def forward(self, h, z, delt_q, dt, hFull):
        h_tem = self.h_Layer(torch.cat([h.float().view(-1,1),
                                        delt_q.float().view(-1,1),
                                        z.float().view(-1,1),
                                        dt.float().view(-1,1)],axis=1))
        
        amid = torch.clip(h_tem, torch.tensor(np.zeros(hFull.shape)).float().view(-1,1), hFull.float().view(-1,1))
        return amid
