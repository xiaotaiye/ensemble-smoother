# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:33:38 2017

@author: xiao
"""
import os
import shutil
import numpy as np
import scipy as sp
import re
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import plotly
import seaborn as sns


#read input.dat
def ReadData():
    f=open('input.dat','r')
    for i in range(4):
        b=f.readline()
    a=f.readline()   
    starttime=float(a[0:3])
    stoptime=float(a[5:])
    global NumDobs,SingleNumDobs,Tstep,Ttime
    Tstep=10
    Ttime=200
    if os.path.exists('ave.dat'):
        os.remove("ave.dat")
    
    b=f.readline()
    a=f.readline() 
    d=re.split(' |\n',a)
    global NxD,NyD,NzD,maxNe,maxNd,Numact
    #NxD,NyD,NzD网格的个数   maxNe: 模型个数 maxNd 参数个数
    NxD=int(d[0])
    NyD=int(d[1])
    NzD=int(d[2])
    Numact=NxD*NyD*NzD
    b=f.readline()
    a=f.readline()
    maxNe=int(a)
    b=f.readline()
    a=f.readline()
    maxNd=int(a)
    f.close()
    #NumDobs=Tstep*maxNd
    for i in range(1,maxNe+1):
        if (i<10):
            shutil.copy2("./temp/permx0"+str(i)+".dat","permx0"+str(i)+".dat")
        else: 
            shutil.copy2("./temp/permx"+str(i)+".dat","permx"+str(i)+".dat")
    
    
#read eclipse summary   *.f file
def ReadOutData(number,ii):
    for j in range(ii):
        k=j+1
        st='base.A00'
        if k<10:
            st=st+'0'+str(k)
        else:
            st=st+str(k)
        f=open(st,'r')
        num=0
        dpredict=np.arange(NumDobs,dtype=float)
        temptime=0.0
        while num<200:
            st=f.readline()
            if len(st)>0 and st[2]=="P":
                st=f.readline()
                d=re.split(' |\n',st)
                temptime=float(d[3]);
                tempyear=float(d[6]);
                dpredict[0]=float(d[9])
                dpredict[1]=float(d[12])
                for i in range(3):
                    st=f.readline()
                    d=re.split(' |\n',st)
                    dpredict[i*4+2]=float(d[3])
                    dpredict[i*4+3]=float(d[6])
                    dpredict[i*4+4]=float(d[9])
                    dpredict[i*4+5]=float(d[12])
                st=f.readline()
                d=re.split(' |\n',st)
                dpredict[14]=float(d[3])
                dpredict[15]=float(d[6])
            num+=1;
            tmp=int(temptime)
            if (tmp>0 and tmp%200==0):
                f.close()
                break
        for i in range(maxNd):
            dpre[number-1,i]=dpredict[i]

def AddNoise():
    global dobscov,dobsmean,dobs,aa
    dobs=np.zeros(NumDobs,dtype=float)
    dobscov=np.zeros((NumDobs,NumDobs),dtype=float)
    aa=(np.random.rand(NumDobs,1)-0.5)
    for i in range(5):
        aa[i]=aa[i]/50
    for i in range(NumDobs):
        dobs[i]=dpre[62,i]+aa[i]
    for i in range(NumDobs):
        dobscov[i,i]=np.fabs(aa[i])

def LogTransfer():
    global lnkU,lnkL,kU,KL
    lnkU=9.213
    lnkL=2.3026
    kL=10.0
    kU=10000.0
    global permx
    for i in range(maxNe):
        for j in range(Numact):
            permx[i,j]=kL if permx[i,j]<kL else permx[i,j]
            permx[i,j]=kU if permx[i,j]>kU else permx[i,j]
    permx=np.log(permx)
    
def LogTransfer2():
    global permx
    for i in range(maxNe):
        for j in range(Numact):
            permx[i,j]=lnkL if permx[i,j]<lnkL else permx[i,j]
            permx[i,j]=lnkU if permx[i,j]>lnkU else permx[i,j]
    permx=np.exp(permx)        

def Work():
    global dmean,dcov,dmmean
    dmean=np.mean(dpre,axis=0)
    print(len(dmean))
    dcov=np.zeros((NumDobs,NumDobs),dtype=float)
    global dd
    dd=np.zeros((maxNe,NumDobs),dtype=float)
    for i in range(maxNe):
        for j in range(NumDobs):
            dd[i,j]=dpre[i,j]-dmean[j]
    for i in range(maxNe):
        dcov=dcov+np.outer(dd[i],dd[i])
    dcov=dcov/(maxNe-1)#模型观测数据的协方差矩阵
    #dcov=np.linalg.inv(dcov)
    dcov=np.linalg.inv(dcov+dobscov)
    global dm,dmmean,dmcov
    dmmean=np.mean(permx,axis=0)
    dm=np.zeros((maxNe,Numact),dtype=float)
    dmcov=np.zeros((Numact,NumDobs),dtype=float)
    for i in range(maxNe):
        for j in range(Numact):
            dm[i][j]=permx[i,j]-dmmean[j]
    #for j in range(Numact):
        #for k in range(NumDobs):
            #for i in range(maxNe):
                #dmcov[j,k]=dmcov[j,k]+(dm[i,j]*dd[i,k])
    for i in range(maxNe):
        dmcov=dmcov=np.outer(dm[i],dd[i])
    dmcov=dmcov/(maxNe-1)#模型参数与观测数据的协方差矩阵
    global temp,test
    test=np.zeros(NumDobs,dtype=float)
    for i in range(maxNe):
        for j in range(NumDobs):
            test[j]=dpre[i,j]-dobs[j]
        temp=np.dot(np.dot(dmcov,dcov),test)
        permx[i,:]=permx[i,:]+temp
    
            

def Init(step):
    global permx
    permx=np.zeros((maxNe,Numact),dtype=float)
    global dpre,NumDobs
    #for ii in range(1):
    NumDobs=maxNd
    dpre=np.zeros((maxNe,NumDobs),dtype=float)
    for i in range(maxNe):
        j=i+1
        if (j<10):
            f=open('permx0'+str(j)+'.dat')
        else: 
            f=open('permx'+str(j)+'.dat')
        f2=open('permx.dat','w')
        a=f.readline()
        f2.write('PERMX\n')
        for jj in range(Numact):
            a=f.readline()
            permx[i,jj]=float(a)
            s=str(permx[i][jj]);
            s=s+'\n';
            f2.write(s)
        f.close()
        f2.write('/')
        f2.close()
        f=open('firstsolution.dat')
        a=f.read()
        f.close()
        f=open('solution.dat','w')
        f.write(a)
        f.close()
        f=open('changeschedule.dat')
        f2=open('schedule.dat','w')
        for jj in range(13):
            a=f.readline()
            f2.write(a)
        f.close()
        a=str(step+1)+'*200 /'
        f2.write(a)
        f2.close()
        os.system('c:\\ecl\\2010.1\\bin\\pc_x86_64\\eclipse base')  
        ReadOutData(j,step)
            #a="base.A0001"
            #os.remove(a)

def WritePermx():
    global permx
    for i in range(maxNe):
        j=i+1
        if (j<10):
            f=open('permx0'+str(j)+'.dat','w')
        else:
            f=open('permx'+str(j)+'.dat','w')
        f.write('PERMX\n')
        for jj in range(Numact):
            s=str(permx[i][jj])
            s=s+'\n'
            f.write(s)
        f.write('/')
        f.close()

def Draw(step):
    global PermxMean,tmp
    PremxMean=np.mean(permx,axis=0)
    f=open('perm'+str(step)+'.dat','w')
    f.write('PERMX\n')
    for jj in range(Numact):
        s=str(PremxMean[jj])
        s=s+'\n'
        f.write(s)
    f.write('/')
    f.close()
    f=open('ave.dat','a+')
    f.write('')
    for i in range(maxNd):
        a=str(dmean[i])+'\n'
        f.write(a)
    f.close()
        
def WriteDobs():
    f=open('measurement.dat','a+')
    for i in range(NumDobs):
        a=str(dobs[i])+'\n'
        f.write(a)
    f.close()
    

def main():
    ReadData()
    for step in range(1,11):
        Init(step)
        AddNoise()
        LogTransfer()
        Work()
        LogTransfer2()
        WritePermx()
        Draw(step)
        WriteDobs()
    #Draw()
    

if __name__ == '__main__':
    main()