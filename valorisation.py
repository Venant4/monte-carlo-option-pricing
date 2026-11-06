import numpy as np
from scipy.stats import norm

def Black_sholes(S0,K,r,vol,T):
    d1=(np.log(S0/K) + (r+0.5*vol**2)*T)/(vol*np.sqrt(T))
    d2=d1 - vol*np.sqrt(T)
    c=S0*norm.cdf(d1) - K *np.exp(-r*T)*norm.cdf(d2)
    p=K *np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    return c,p

def Wiener(T,paths,N):
    W=np.zeros((paths,N+1))
    dt=T/N
    Z=np.random.normal(0,1,size=(paths,N))
    W[:,0]=0
    for i in range(paths):
        for j in range(N):
            W[i,j+1]=W[i,j] + Z[i,j]*np.sqrt(dt)
    return W

def Montecarlo(S0,r,vol,T,paths,N):
    ST=np.zeros(paths)
    W=Wiener(T,paths,N)
    WT=W[:,-1]
    for i in range(paths):
        ST[i]=S0*np.exp((r-0.5*vol**2)*T + vol*WT[i])
    return ST

def Euler(S0,r,vol,T,paths,N):
    dt=T/N
    histo=np.zeros((paths,N+1))
    histo2=np.zeros((paths,N+1))
    histo[:,0]=S0
    histo2[:,0]=S0
    Z=np.random.normal(0,1,size=(paths,N))
    for i in range(paths):
        for j in range(N):
            histo[i,j+1]=histo[i,j] + r*dt*histo[i,j] + vol*histo[i,j]*Z[i,j]*np.sqrt(dt)
    return histo

def Euler_ant(S0,r,vol,T,paths,N):
    dt=T/N
    histo=np.zeros((paths,N+1))
    histo2=np.zeros((paths,N+1))
    histo[:,0]=S0
    histo2[:,0]=S0
    Z=np.random.normal(0,1,size=(paths,N))
    Z_ant=-Z
    for i in range(paths):
        for j in range(N):
            histo[i,j+1]=histo[i,j] + r*dt*histo[i,j] + vol*histo[i,j]*Z[i,j]*np.sqrt(dt)
            histo2[i,j+1]=histo2[i,j] + r*dt*histo2[i,j] + vol*histo2[i,j]*Z_ant[i,j]*np.sqrt(dt)
    return histo,histo2


def Pricing_Euler_ant(S0,r,vol,T,paths,N,K):
    histo,histo2=Euler_ant(S0,r,vol,T,paths,N)
    ST1=histo[:,-1]
    ST2=histo2[:,-1]
    payoff1=np.maximum(ST1-K,0)
    payoff2=np.maximum(ST2-K,0)
    payoff=(payoff1 + payoff2)/2
    return np.exp(-r*T)*np.mean(payoff)
  

def pricing_MOntecarlo(S0,r,vol,T,paths,N,K):
    ST=Montecarlo(S0,r,vol,T,paths,N)
    payoff_c=np.maximum(ST-K,0)
    payoff_p=np.maximum(K-ST,0)
    c=np.exp(-r*T)*np.mean(payoff_c)
    p=np.exp(-r*T)*np.mean(payoff_p)
    return c,p

def Pricing_Euler(S0,r,vol,T,paths,N,K):
    histo=Euler(S0,r,vol,T,paths,N)
    ST=histo[:,-1]
    payoff_c=np.maximum(ST-K,0)
    payoff_p=np.maximum(K-ST,0)
    return np.exp(-r*T)*np.mean(payoff_c), np.exp(-r*T)*np.mean(payoff_p)
def Asiatique(S0,r,vol,T,paths,N,K):
    histo=Euler(S0,r,vol,T,paths,N)
    Smoy=np.mean(histo,axis=1)
    payoff_c=np.maximum(Smoy-K,0)
    payoff_p=np.maximum(K-Smoy,0)
    return np.exp(-r*T)*np.mean(payoff_c), np.exp(-r*T)*np.mean(payoff_p)

def DownandOut(S0,r,vol,T,paths,N,K,H):
    histo=Euler(S0,r,vol,T,paths,N)
    ST=histo[:,-1]
    payoffc=np.zeros(paths)
    for i in range(paths):
        Barrier=False
        for j in range(1,N+1):
            if histo[i,j]<=H:
                Barrier=True
        if not Barrier:
            payoffc[i]=np.maximum(ST[i]-K,0)
    return np.exp(-r*T)*np.mean(payoffc)

def UpandIn(S0,r,vol,T,paths,N,K,H):
    histo=Euler(S0,r,vol,T,paths,N)
    ST=histo[:,-1]
    payoffc=np.zeros(paths)
    for i in range(paths):
        barrier=False
        for j in range(1,N+1):
            if histo[i,j]>=H:
                barrier=True
        if barrier:
            payoffc[i]=np.maximum(ST[i]-K,0)
    return np.exp(-r*T)*np.mean(payoffc)

def Digitale(S0,r,vol,T,paths,N,K,Q):
    histo=Euler(S0,r,vol,T,paths,N)
    ST=histo[:,-1]
    payoffc=np.zeros(paths)
    for i in range(paths):
        if ST[i]>K:
            payoffc[i]=Q
        else:
            payoffc[i]=0
    return np.exp(-r*T)*np.mean(payoffc)


class BlackSholes:
    def __init__(self,S0,K,T,sigma,r):
        self.S0=S0
        self.K=K
        self.T=T
        self.sigma=sigma
        self.r=r
    def Prix(self):
        d1=(np.log(self.S0/self.K)+(self.r+0.5*self.sigma**2)*self.T)/(self.sigma*np.sqrt(self.T))
        d2=d1-self.sigma*np.sqrt(self.T)
        c=self.S0*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        p=-self.S0*norm.cdf(-d1) + self.K*np.exp(-self.r*self.T)*norm.cdf(-d2)
        return c,p

class Montecarlo:
    def __init__(self,modele,paths,N):
        self.modele=modele
        self.paths=paths
        self.N=N

    def Wiener(self):
        dt=self.T/self.N
        Z=np.random.normal(0,1,size=(self.paths,self.N))
        W=np.zeros((self.paths,self.N+1))
        W[:,0]=0
        for i in range(self.paths):
            for j in range(self.N):
                W[i,j+1] = W[i,j] + Z[i,j]*np.sqrt(dt)
        return W[:,-1]
    def SolutionExpl(self):
        W=self.Wiener()
        ST=self.modele.S0*np.exp((self.modele.r - 0.5*self.modele.sigma**2)*self.modele.T + self.modele.sigma*W)
        return ST

class Euler:
    def __init__(self,modele,paths,N):
        self.modele=modele
        self.paths=paths
        self.N=N

    def Trajectoire(self):
        histo=np.zeros((self.paths,self.N+1))
        histo[:,0]=self.modele.S0
        dt=self.modele.T/self.N
        Z=np.random.normal(0,1,size=(self.paths,self.N))
        for i in range(self.paths):
            for j in range(self.N):
                histo[i,j+1] = histo[i,j] + self.modele.r*dt*histo[i,j] + self.modele.sigma*histo[i,j]*Z[i,j]*np.sqrt(dt)
        return histo

class Option:
    def __init__(self,modele,eul):
        self.modele=modele
        self.eul=eul
        self.histo=self.eul.Trajectoire()
class Call_vanille(Option):
    def Payoff(self):
        ST=self.histo[:,-1]
        return np.maximum(ST-self.modele.K,0)
    def Prix(self):
        return np.exp(-self.modele.r*self.modele.T)*np.mean(self.Payoff())

class Digitale(Option):
    def __init__(self, modele, eul,Q):
        super().__init__(modele, eul)
        self.Q=Q
    def Payoff(self):
        ST=self.histo[:,-1]
        p=np.zeros(self.mc.paths)
        for i in range(len(p)):
            if ST[i]>self.modele.K:
                p[i]=self.Q
            else:
                p[i]=0
        return p
    def Prix(self):
        return np.exp(-self.modele.r*self.modele.T)*np.mean(self.Payoff())

class Asiatique(Option):
    def Payoff(self):
        Smoy=np.mean(self.histo,axis=1)
        return np.maximum(Smoy-self.modele.K,0)
    def Prix(self):
        return np.exp(-self.modele.r*self.modele.T)*np.mean(self.Payoff())

class UpandOut(Option):
    def __init__(self, modele, eul,H):
        super().__init__(modele, eul)
        self.H=H
    def Payoff(self):
        p=np.zeros(self.eul.paths)
        barrier=False
        ST=self.histo[:,-1]
        for i in range(self.eul.paths):
            for j in range(1,self.eul.N):
                if self.histo[i,j]>=self.H:
                    barrier=True
                    break
            if not barrier:
                p[i]=np.maximum(ST-self.modele.K,0)
        return p
    def Prix(self):
        return np.exp(-self.modele.r*self.modele.T)*np.mean(self.Payoff())

    

    

    
            
            
        


        
    

        



    

