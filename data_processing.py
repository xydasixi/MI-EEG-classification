import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
import scipy.linalg as la
import scipy.io
from scipy.linalg import logm
# import matlab
# import matlab.engine
# engine = matlab.engine.start_matlab() # Start MATLAB process
# def datagenerator(data_path):
#     file_list = glob.glob(data_path + '/*.npz')
#     dataX0 = np.load(file_list[0])['X']
#     dataY0 = np.load(file_list[0])['y']
#     for i in range(len(file_list)-1):
#         tempX = np.load(file_list[i+1])['X']
#         tempY = np.load(file_list[i+1])['y']
#         dataX0 = np.vstack((dataX0,tempX))
#         dataY0 = np.hstack((dataY0,tempY))
#     dataX = dataX0.reshape(-1,30,25).astype('float32')
#     dataX = np.expand_dims(dataX,axis=1)
#     dataY = np.zeros(len(dataX))
#     for i in range(len(dataY)):
#         dataY[i] = dataY0[i//len(dataX0[0])]
#     dataY = dataY.astype('int64')
#     data = TensorDataset(dataX,dataY)
#     return data

class EA:
    def __init__(self,src):
        self.src = src
        self.dest = 1

def spd(A):#输入对称正定矩阵A
    eig_val, eig_vec = eig(A) #特征分解
    eig_diag = np.diag(1/(eig_val**0.5)) #特征值开方取倒数对角化
    B = np.dot(np.dot(eig_vec, eig_diag), inv(eig_vec)) #inv为求逆
    return B #返回A的-1/2次幂

def EA(src):
    [t,c,p]=src.shape #trails，通道数，采样点数
    dest = src.copy()#对齐后的数据
    R = np.zeros((c,c))#协方差矩阵
    for i in range(t):
        R += np.dot(src[i],np.transpose(src[i]))
    R = R/t
    R_1= spd(R)
    for i in range(t):
        dest[i] = np.dot(R_1,src[i])
    return dest

class CSP:
    def __init__(self,src1,src2,feature_num):
        self.feature_num = feature_num #可选择的前m行和后m行（2m<M）作为原始输入数据的特征
        [self.t,self.c,self.p]=src1.shape #trails，通道数，采样点数
        self.R1,self.R2,self.R = self.mean_covariance(src1,src2)
        self.P = self.eigenvalue()
        self.W = self.make_filter()
        self.f1,self.f2 = self.feature_vector(src1,src2)

    #计算两类信号的特征向量
    def feature_vector(self,src1,src2):
        [t, c] = [self.t, self.c]
        W = self.W
        m = self.feature_num
        W = np.vstack((W[0:m],W[-m:]))
        f1 = np.zeros((t,2*m)) #一类数据的特征
        f2 = np.zeros((t, 2 * m)) #二类数据的特征
        for i in range(t):
            Z1 = np.dot(W,src1[i])
            Z2 = np.dot(W,src2[i])
            var1 = np.var(Z1,axis = 1)
            var2 = np.var(Z2, axis=1)
            f1[i] = var1 / np.sum(var1)
            f2[i] = var2/ np.sum(var2)
        return f1,f2


    #空间滤波器
    def make_filter(self):
        R1,R2,P = self.R1,self.R2,self.P
        S1 = np.dot(np.dot(P, R1), np.transpose(P))
        S2 = np.dot(np.dot(P, R2), np.transpose(P))
        lambda1,B1 = self.EVD(S1)
        lambda2, B2 = self.EVD(S2)
        W =np.dot(np.transpose(B1),P)
        return W

    #计算白化特征值矩阵
    def eigenvalue(self):
        R = self.R
        # scipy.io.savemat('result.mat', {'key1':R})
        e_vals, e_vecs = self.EVD(R)
        k = np.sqrt(la.inv(e_vals))
        P = np.dot(spd(e_vals),np.transpose(e_vecs))
        return P

    # 特征值分解
    def EVD(self,src):
        e_vals, e_vecs = np.linalg.eigh(src)
        idx = np.argsort(-e_vals)
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        e_vals = np.diag(e_vals)
        # src = matlab.double(src.tolist())
        # e_vecs, e_vals = engine.EIG(src,nargout=2) # 特征值，特征矩阵
        # e_vals= np.array(e_vals)
        # e_vecs= np.array(e_vecs)
        return e_vals,e_vecs

    #计算平均协方差矩阵
    def mean_covariance(self,src1,src2):
        [t,c] = [self.t,self.c]
        R1 = np.zeros((c,c))
        R2 = np.zeros((c,c))
        for i in range(t):
            R1 += self.covariance(src1[i])
            R2 += self.covariance(src2[i])
        R1 /= t
        R2 /= t
        R = np.add(R1,R2)
        return R1,R2,R
    #计算协方差
    def covariance(self,src):
        dest = np.dot(src,np.transpose(src))
        trace = np.trace(dest)
        dest = dest/trace
        return dest



if __name__ == '__main__':
    dataX = np.load('data/train/S1.npz')['X']
    dataY = np.load('data/train/S1.npz')['y']
    [o,p,q] = dataX.shape
    # sb = np.zeros((q,p,o))
    # for i in range(q):
    #     for j in range(p):
    #         for k in range(o):
    #             sb[i][j][k] = dataX[k][j][i]
    # scipy.io.savemat('result.mat', {'key1':sb})
    dest = EA(dataX)
    dest = dest * 10
    dataX0 =[]
    dataX1 =[]
    for i in range (len(dataY)):
        if dataY[i] == 0:
            dataX0.append(dest[i])
        else:
            dataX1.append(dest[i])
    dataX0 = (np.array(dataX0)).astype('float32')
    dataX1 = (np.array(dataX1)).astype('float32')

    f1 = CSP(dataX0, dataX1,1).f1
    f2 = CSP(dataX0, dataX1, 1).f2

    plt.scatter(f1[:,0],f1[:,1])
    plt.scatter(f2[:, 0], f2[:, 1])
    plt.show()
    c = 0