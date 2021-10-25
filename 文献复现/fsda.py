import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

#数据输入
f=open('E:\论文实现\数据\phoneme.data')
df=pd.read_csv(f)
df=df.drop('row.names',1)

#构造关于调优参数m,h的函数，返回50次随机划分数据的测试误差
def acc_test(m,h):
    acc_test=np.ones(50)  #初始化误差
    for j in range(50):   #循环50次
        # 随机选取1000个样本作为训练集，剩余3509个为测试集
        index=np.random.randint(0,4509,1000)
        df_train=df.iloc[index,]
        df_test=df.drop(index)
        data_train=np.array(df_train)
        data_test=np.array(df_test)

        #将数据按类标签分成五类
        df1=df_train.query('g=="sh"')
        df2=df_train.query('g=="dcl"')
        df3=df_train.query('g=="iy"')
        df4=df_train.query('g=="aa"')
        df5=df_train.query('g=="ao"')

        #划分自变量与因变量
        X_train=data_train[:,0:256]
        X_test=data_test[:,0:256]
        Y_train=data_train[:,256]
        Y_test=data_test[:,256]
        X_sh=np.array(df1)[:,0:256]
        X_dcl=np.array(df2)[:,0:256]
        X_iy=np.array(df3)[:,0:256]
        X_aa=np.array(df4)[:,0:256]
        X_ao=np.array(df5)[:,0:256]

        # 计算F统计量，选出m个h-separared markers
        X_mean = np.average(X_train, 0)
        X_sh_mean = np.average(X_sh, 0)
        X_dcl_mean = np.average(X_dcl, 0)
        X_iy_mean = np.average(X_iy, 0)
        X_aa_mean = np.average(X_aa, 0)
        X_ao_mean = np.average(X_ao, 0)
        # n1=len(X_sh)
        # n2=len(X_dcl)
        # n3=len(X_iy)
        # n4=len(X_aa)
        # n5=len(X_ao)
        Sb = (len(X_sh) * np.square(X_sh_mean - X_mean) + len(X_dcl) * np.square(X_dcl_mean - X_mean)
              + len(X_iy) * np.square(X_iy_mean - X_mean) + len(X_aa) * np.square(X_aa_mean - X_mean)
              + len(X_ao) * np.square(X_ao_mean - X_mean)) / 4
        Sw = (np.var(X_sh, axis=0) * len(X_sh) + np.var(X_dcl, axis=0) * len(X_dcl)
              + np.var(X_iy, axis=0) * len(X_iy) + len(X_aa) * np.var(X_aa, axis=0)
              + len(X_ao) * np.var(X_ao, axis=0)) / (len(X_train) - 5)
        F = Sb / Sw
        markers = findmarkers(m, h, F)


        # LDA降维，得到3m个线性判别变量
        lda = LinearDiscriminantAnalysis(n_components=3)
        lda.fit(X_train[:, int(max(markers[0] - h, 0)):int(min(markers[0] + h, X_train.shape[1] - 1))], Y_train)
        X_train_ldv = lda.transform(
            X_train[:, int(max(markers[0] - h, 0)):int(min(markers[0] + h, X_train.shape[1] - 1))])
        X_test_ldv = lda.transform(X_test[:, int(max(markers[0] - h, 0)):int(min(markers[0] + h, X_test.shape[1] - 1))])
        for i in range(1, m):
            lda.fit(X_train[:, int(max(markers[i] - h, 0)):int(min(markers[i] + h, X_train.shape[1] - 1))], Y_train)
            X_train_new = lda.transform(
                X_train[:, int(max(markers[i] - h, 0)):int(min(markers[i] + h, X_train.shape[1] - 1))])
            X_test_new = lda.transform(
                X_test[:, int(max(markers[i] - h, 0)):int(min(markers[i] + h, X_test.shape[1] - 1))])
            X_train_ldv = np.hstack((X_train_ldv, X_train_new))
            X_test_ldv = np.hstack((X_test_ldv, X_test_new))

        #SVM分类
        clf = SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(X_train_ldv, Y_train)
        # print(clf.score(X_train_ldv, Y_train))
        acc_test[j]=clf.score(X_test_ldv, Y_test)
    #print(acc_test)
    acc_test_mean=np.average(acc_test)
    #print(acc_test_mean)  #输出测试误差
    return acc_test_mean

#定义寻找markers的函数
def findmarkers(m,h,F):
    markers=np.zeros(m)
    for i in range(m):
        markers[i]=np.argmax(F)
        F[max(np.argmax(F)-h,0):min(np.argmax(F)+h,len(F)-1)]=0
    return markers

print('测试集平均准确率为:%.10f'%(acc_test(7,6)))


