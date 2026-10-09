import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc, confusion_matrix


def check(y,d):
    y=np.asarray(y);d=np.asarray(d,dtype=float)
    if len(y)!=len(d) or set(np.unique(y))!={0,1} or not np.isfinite(d).all() or (d<0).any():
        raise ValueError('Need finite nonnegative distances and both binary labels')
    return y,d


def select_threshold(y,d):
    y,d=check(y,d)
    far,tpr,score_threshold=roc_curve(y,-d,drop_intermediate=False)
    frr=1-tpr
    valid=np.isfinite(score_threshold)
    ids=np.flatnonzero(valid)
    i=ids[np.argmin(np.abs(far[valid]-frr[valid]))]
    return float(-score_threshold[i])


def metrics(y,d,threshold):
    y,d=check(y,d);pred=(d<=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    far,tpr,_=roc_curve(y,-d,drop_intermediate=False);frr=1-tpr
    i=int(np.argmin(np.abs(far-frr)))
    precision,recall,_=precision_recall_curve(y,-d)
    return dict(n=len(y),threshold=float(threshold),accuracy=float((tp+tn)/len(y)),
        precision=float(tp/(tp+fp)) if tp+fp else 0.,recall=float(tp/(tp+fn)),
        f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
        specificity=float(tn/(tn+fp)),sensitivity=float(tp/(tp+fn)),
        FAR=float(fp/(fp+tn)),FRR=float(fn/(fn+tp)),
        EER=float((far[i]+frr[i])/2),EER_method='nearest empirical ROC operating point; diagnostic, not deployment threshold',
        ROC_AUC=float(roc_auc_score(y,-d)),PR_AUC=float(auc(recall,precision)),
        average_precision=float(average_precision_score(y,-d)),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])
