import pandas as pd

class Nodes:
    def __init__(self,nodes_id,nodes_z,nodes_hfull):
        self.nodes_id = nodes_id
        self.nodes_full = {}
        self.nodes_z = {}
        self.nodes_h = {}
        for it in nodes_id:
            self.nodes_z[it] = nodes_z[it]
            self.nodes_h[it] = 0.0
            self.nodes_full[it] = nodes_hfull[it]
    
    def set_h(self,h):
        # update h every time after simulation
        for it in self.nodes_id:
            self.nodes_h[it] = h[it]
            
    def save_h(self,filepath):
        # save h as csv
        pd.DataFrame(self.nodes_h).to_csv('h_'+filepath)
        

class Links:
    def __init__(self,links_id,xsects):
        self.links_id = links_id
        self.xsect = {}
        self.links_q = {}
        self.links_amid = {}
        for it in links_id:
            self.links_q[it] = 0
            self.links_amid[it] = 0
            self.xsect[it] = xsects[it]
            
    def set_qa(self,q,a):
        # update q and amid every time after simulation
        for it in self.links_id:
            self.links_q[it] = q[it]
            self.links_amid[it] = a[it]
            
    def save_qa(self,filepath):
        # save qa as csv
        pd.DataFrame(self.links_q).to_csv('q_'+filepath)
        pd.DataFrame(self.links_amid).to_csv('a_'+filepath)


class section_single:
    def __init__(self,yfull,afull,rfull,length,sigma):
        self.yFull = yfull
        self.rFull = rfull
        self.aFull = afull
        self.length = length
        self.sigma = sigma

class Pipe_Network:
    def __init__(self,links_id,links_xsects,nodes_id,nodes_z,nodes_hfull,connection,node_connection):
        self.links = Links(links_id,links_xsects)
        self.nodes = Nodes(nodes_id,nodes_z,nodes_hfull)
        self.connection = connection
        self.node_connection = node_connection
        
