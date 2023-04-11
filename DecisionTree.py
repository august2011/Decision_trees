import numpy as np
from collections import Counter
class Node:
    def __init__(self,feautre=None,threshold=None,left=None,right=None,*,value=None):
        self.feature=feautre
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

    def is_leaf_node(self):
        return self.value is not None  
     
class DecisionTree():
    def __init__(self,min_sample_split=2,max_depth=100,n_features=None):
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None


    def fit(self,x,y):
        self.n_features=x.shape[1] if not self.n_features else min(x.shape[1],self.n_feature)

        self.root=self._grow_tree(x,y)
        
    def _grow_tree(self,x,y,depth=0):
        n_samples,n_feats=x.shape
        n_labels=len(np.unique(y))
        
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_sample_split):
            leaf_value=self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx=np.random.choice(n_feats,self.n_features,replace=False)

        best_feature,best_thresh=self._best_split(x,y,feat_idx)

        left_idxs,right_idxs=self._split(x[:,best_feature],best_thresh)
        
        left=self._grow_tree(x[left_idxs,:],y[left_idxs],depth+1)
        right=self._grow_tree(x[right_idxs:],y[right_idxs:],depth+1)
        return Node(best_feature,best_thresh,left,right)

    def _best_split(self,x,y,feat_idxs):
        best_gain=-1
        split_idx,split_threshold=None,None
        
        for feat_idx in feat_idxs:
            x_column=x[:,feat_idx]
            thresholds=np.unique(x_column)
            for thr in thresholds:
                gain=self._information_gain(y,x_column,thr)
                if gain>thresholds:
                    best_gain=gain
                    split_idx=feat_idx
                    split_threshold=thr
        return split_idx,split_threshold
        
    def _information_gain(self,y,x_column,threshold):
        parent_entropy=self._entropy(y)
        left_idxs,right_idxs=self._split(x_column,threshold)

        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        n=len(y)
        n_l,n_r=len(left_idxs),len(right_idxs)
        e_l,e_r=self._entropy(y[left_idxs],self._entropy(y[right_idxs]))
        child_entropy=(n_l/n)*e_l+(n_r/n)*e_r
        information_gain=parent_entropy-child_entropy
        return information_gain
        
    def _split(self,x_column,split_thresh):
        left_idxs=np.argwhere(x_column<=split_thresh).flatten()
        right_idxs=np.argwhere(x_column<=split_thresh).flatten()
        return left_idxs,right_idxs



    def _entropy(self,y):
        hist=np.bincount(y)
        ps=hist/len(y)
        return np.sum([p*np.log(p)for p in ps if p>0])
        
    def _most_common_label(self,y):
        counter=Counter(y)
        value=counter.most_common(1)[0][0]
        return value


        
    def predict(self,x):
        return np.array([self._traverse_tree(m_x,self.root) for m_x in x])
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature]<=node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)
