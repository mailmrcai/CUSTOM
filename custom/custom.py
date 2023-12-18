__author__ = "Xavier Hernandez-Alias"
'''
Module for optimizing the codon usage of a sequence based on tissue 
specificities. 
'''
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
import time
import pkg_resources
import numpy as np
import pandas as pd
import RNA
from sympy import N

# Load data required for optimization
GENETIC_CODE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'}

def load_data(file):
    '''
    Return a dataframe of the required file.
    '''
    stream = pkg_resources.resource_stream(__name__, file)
    return pd.read_csv(stream, index_col=0)

codon_weights = load_data('data/CUSTOM_codon_weights.csv')
codon_ratios = load_data('data/CUSTOM_tissue_ratios.csv')
codon_freq = load_data('data/CUSTOM_codonfreq_CoCoPuts.csv')
codpair_freq = load_data('data/CUSTOM_codonpairsfreq_CoCoPuts.csv')


def relative_codons():
    '''
    Computes normalized codon abundances for each amino-acid family.

    Returns
    -------
    codnorm: DataFrame

    '''
    if "codnorm" not in globals():
        global codnorm
        codnorm = pd.DataFrame()
        AAs = list(set(GENETIC_CODE.values())); AAs.remove("_")
        for species in codon_freq.columns:
            for aa in AAs:
                cod_aa = [c for c in GENETIC_CODE.keys() if GENETIC_CODE[c]==aa]
                max_aa = codon_freq.loc[cod_aa,species].max()
                for cod in cod_aa:
                    codnorm.loc[cod,species] = codon_freq.loc[cod,species]/max_aa

def compute_CPS():
    '''
    Computes the Codon Pair Scores based on the observed vs expected
    frequency. The CPS is + for overrepresented pairs and - for 
    underrepresented pairs. The expected frequency is computed to be 
    independent both of amino acid frequency and of codon bias.

    Returns
    -------
    CPSs: DataFrame

    '''
    if "CPSs" not in globals():
        global CPSs
        CPSs = pd.DataFrame()
        for species in codpair_freq.columns:
            for pair in codpair_freq.index:
                if (pair[0:3] not in ["TAA","TGA","TAG"]) and (pair[3:6] not in ["TAA","TGA","TAG"]):
                    obs_pair = codpair_freq.loc[pair,species]
                    # Compute codon occurrences
                    cod_A = codon_freq.loc[pair[0:3],species]
                    cod_B = codon_freq.loc[pair[3:6],species]
                    # Compute AA occurrences
                    aaA_cods = [c for c in GENETIC_CODE.keys() if GENETIC_CODE[c]==GENETIC_CODE[pair[0:3]]]
                    aaB_cods = [c for c in GENETIC_CODE.keys() if GENETIC_CODE[c]==GENETIC_CODE[pair[3:6]]]
                    aa_A = codon_freq.loc[aaA_cods,species].sum()
                    aa_B = codon_freq.loc[aaB_cods,species].sum()
                    # Compute AA pair occurences
                    aaAB_codpairs = [c1+c2 for c1 in aaA_cods for c2 in aaB_cods]
                    aa_AB = codpair_freq.loc[aaAB_codpairs,species].sum()
                    # Compute expected frequency
                    exp_pair = ((cod_A*cod_B)/(aa_A*aa_B))*aa_AB
                    # Compute CPS
                    CPSs.loc[pair,species] = np.log(obs_pair/exp_pair)

def action(metric,to_do):
    '''
    Takes a list of computed values and converts them to minimize or maximize

    Parameters
    ----------
    metric : ndarray
        Values to optimize
    
    to_do : {"min","max"}
        Min or Max

    Returns
    -------
    norm : ndarray

    '''
    if to_do=="max":
        norm = metric
        return norm
    elif to_do=="min":
        norm = -metric
        return norm
    else:
        raise TypeError("Invalid 'by' argument. Please specify either 'max' "
                        "or 'min' to indicate whether each "
                        "metric should be maximized or minimized.")


def optimize_single_codon(aa,prob_dict):
    codons = list(prob_dict[aa].keys())
    newcodon = ""
    while not newcodon:
        if len(codons)==1:
            newcodon = codons[0]
        else:
            probs = np.array([prob_dict[aa][c][0] for c in codons])
            probs = probs/np.sum(probs)
            action = np.random.choice(codons,p=probs)
            if prob_dict[aa][action][1]==True:
                newcodon = action
            elif prob_dict[aa][action][1]==False:
                codons.remove(action)
            else:
                if np.random.uniform()<=0.5:
                    newcodon = action
                else:
                    codons.remove(action)
    return newcodon

def optimize_single_seq(sequence,prob_dict,prob_original):
    seqcodons = [sequence[n:n+3] for n in range(0,len(sequence),3)]
    # Create pool of optimized sequences
    finalseq = ""
    # for n in range(self.n_pool):
    # while len(finalseq) != len(sequence):
        # for n,codon in enumerate(seqcodons):
    n = 0
    # print("final_seq",finalseq)
    while n < len(seqcodons):
        # prob_original allows to take a conservative optimization if wanted
        codon = seqcodons[n]
        if codon in ["ATG","TGG","TAA","TGA","TAG"]: # do not touch stop codons nor ATG or TGG
            finalseq += codon
        elif np.random.uniform() >= prob_original:
            aa = GENETIC_CODE[codon]
            # newseq.append(optaa(self,aa))
            finalseq += optimize_single_codon(aa,prob_dict)
        else:
            finalseq += codon
        if any(i in finalseq[-9:] for i in ["AAAAAAA","GGGGGGG","TTTTTTT","CCCCCCC"]):
            finalseq = finalseq[:-9]
            n = n - 3
        n = n + 1
    return finalseq


def get_data_in_pool(func,pool):
    thread_pool = ProcessPoolExecutor(40)
    task_list = []
    for index,seq in enumerate(pool):
        task_list.append(thread_pool.submit(func,index,seq))
    result_list = []
    for task in as_completed(task_list):
        result_list.append(task.result())
    result_list = sorted(result_list,key=lambda x:x[0])
    return [i[1] for i in result_list]


def get_MFE(index,seq):
    windows = [seq[w:w+40] for w in range(40,(len(seq)-39))]
    mfe_temp = []
    for w in windows:
        mfe_temp.append(RNA.fold(w)[1])
    return index,np.mean(mfe_temp)

def get_MFEini(index,seq):
    return index,RNA.fold(seq[:40])[1]


def get_CAI(index,seq):
    seqcodons = [seq[n:n+3] for n in range(0,len(seq),3)]
    codus = {c:seqcodons.count(c) for c in set(seqcodons) if c not in ["TAA","TGA","TAG"]}
    cai_codus = [(codnorm.loc[c,"Homo_sapiens"]**(1/sum(codus.values())))**codus[c] for c in codus.keys()]
    return index,np.prod(cai_codus)




def get_CPB(index,seq):
    seqcodpairs = [seq[n:n+6] for n in range(0,len(seq),3) if (len(seq[n:n+6])==6)and(seq[n+3:n+6] not in ["TAA","TGA","TAG"])]
    # codscores = [CPSs.loc[pair,"Homo_sapiens"] for pair in seqcodpairs]
    codscores = []
    for pair in seqcodpairs:
        try:
            codscores.append(CPSs.loc[pair,"Homo_sapiens"])
        except:
            ...
    return index,np.mean(codscores)

def get_ENC(index,seq):
    codon_families = {aa:list(GENETIC_CODE.values()).count(aa) for aa in GENETIC_CODE.values() if aa!="_"}
    seqcodons = [seq[n:n+3] for n in range(0,len(seq),3)]
    codus = {c:seqcodons.count(c) for c in set(seqcodons) if c not in ["TAA","TGA","TAG"]}
    ENC_family = {}
    for f in [2,3,4,6]:
        fam_aa = [aa for aa in codon_families.keys() if codon_families[aa]==f]
        ENCfam = []
        for aa in fam_aa:
            cod_aa = [c for c in GENETIC_CODE.keys() if GENETIC_CODE[c]==aa]
            if any([c in codus.keys() for c in cod_aa]):
                codus_family = np.array([codus[c] if c in codus.keys() else 0.0 for c in cod_aa])
                codus_sum = codus_family.sum()
                if codus_sum>1:
                    p_family = codus_family/codus_sum
                    numerator = (codus_sum - 1.0)
                    denominator = (codus_sum * np.sum([p**2 for p in p_family]) - 1.0)
                    if numerator!=0 and denominator!=0:
                        ENCfam.append(numerator/denominator)
        ENC_family[f] = ENCfam
    # Compute contributions of each family
    enc_codus = 2.0 + 9.0*np.mean(ENC_family[2]) + 1.0*np.mean(ENC_family[3]) + 5.0*np.mean(ENC_family[4]) + 3.0*np.mean(ENC_family[6])
    return index,enc_codus

def get_GC(index,seq):
    if len(seq) == 0:
        return index,0
    return index,(seq.count("G")+seq.count("C"))/len(seq)


class TissueOptimizer:
    '''
    Optimize object, which contains all required methods for tissue-optimizing 
    the codons an amino-acid or nucleotide sequence. Codons are selected based 
    on the Random Forest features that define tissue-specificities, which
    are considered as the probability of optimizing each codon. Directionality
    is based on tissue-specific SDA ratios. Among the pool of generated 
    sequences, the best ones are selected based several commonly used metrics
    (CAI, ENC, CPB, MFE, etc.).
    
    Parameters
    ----------
    tissue: {"Lung","Breast","Skin","Spleen","Heart","Liver","Salivarygland",
             "Muscle...Skeletal","Tonsil","Smallintestine","Placenta",
             "Appendices","Testis","Rectum","Urinarybladder","Prostate",
             "Esophagus","Kidney","Thyroid","Lymphnode","Artery","Brain",
             "Nerve...Tibial","Gallbladder","Uterus","Pituitary","Colon",
             "Vagina","Duodenum","Fat","Stomach","Adrenal","Fallopiantube",
             "Smoothmuscle","Pancreas","Ovary"}
        Tissue to which optimize the sequence
    
    n_pool: int, default=100
        The number of sequences in the generated pool of optimized sequences.
        
    degree: float between 0-1, default=0.5
        Percentage of codons to optimize. Higher values lead to optimizing 
        all codons. Lower values optimize only the most clearly 
        tissue-specific codons.
    
    prob_original: float between 0-1, default=0.0
        Extent to which original codons are conserved. Higher values lead to 
        more conservative optimizations.
    
    Attributes
    ----------
    pool : list
        Minimum Free Energy of optimized sequences. Predicted secondary 
        structures are based on the Vienna RNA Package (Lorenz et al., 2011).
    
    codonprob : dict
        For each amino acid, it contains a nested dictionary matching each
        codon with (1) its tissue-specific weight and (2) its directionality
        (True if its inclusion is favored in that tissue, False otherwise).
    
    sequence : str
        Original sequence to be optimized.
    
    Examples
    --------
    >>> import custom
    >>> opt = TissueOptimizer("kidney", n_pool=50)
    >>> seq = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSG"
    >>> opt.optimize(seq)
    >>> best_seq = opt.select_best(top=10)
    
    '''
    def __init__(self, tissue, n_pool=100, degree=0.5, prob_original=0.0):
        if tissue in codon_weights.index:
            self.tissue = tissue
            self.n_pool = n_pool
            self.degree = degree
            self.prob_original = prob_original
            # Get directionality and probabilities for tissue optimization
            # "degree" determines to which extent are unclear codons optimized
        else:
            raise TypeError("Invalid 'tissue' argument. Allowed tissues are: "
                            "Lung, Breast, Skin, Spleen, Heart, Liver, Salivarygland, "
                            "Muscle...Skeletal, Tonsil, Smallintestine, Placenta, "
                            "Appendices, Testis, Rectum, Urinarybladder, Prostate, "
                            "Esophagus, Kidney, Thyroid, Lymphnode, Artery, Brain, "
                            "Nerve...Tibial, Gallbladder, Uterus, Pituitary, Colon, "
                            "Vagina, Duodenum, Fat, Stomach, Adrenal, Fallopiantube, "
                            "Smoothmuscle, Pancreas, Ovary.")
    def caculate_codonprob(self):
        threshold = np.percentile(codon_ratios.loc[self.tissue,:].abs(),(1.0-self.degree)*100)
        codonprob = {}
        for aa in set(GENETIC_CODE.values()):
            aa_codons = [c for c in GENETIC_CODE.keys() if np.logical_and((GENETIC_CODE[c] is aa),(c not in ["ATG","TGG","TAA","TGA","TAG"]))]
            # Avoid deterministic behaviour of codons with weight=0
            if any(codon_weights.loc[self.tissue,aa_codons]):
                weight_bkg = codon_weights.loc[self.tissue,aa_codons].mean()
                codweights = {c:codon_weights.loc[self.tissue,c]+weight_bkg for c in aa_codons}
            else:
                codweights = {c:codon_weights.loc[self.tissue,c] for c in aa_codons}
            sumweights = sum(codweights.values())
            if sumweights==0:
                sumweights = 1.0
            coddirection = {c:codon_ratios.loc[self.tissue,c]>0 if np.abs(codon_ratios.loc[self.tissue,c])>threshold else np.nan for c in aa_codons}
            codonprob[aa] = {c:[codweights[c]/sumweights,coddirection[c]] for c in aa_codons}
        # self.codonprob = codonprob
        return codonprob
    

    def get_optimized_seq_pool(self, sequence):
        # print(optimize_single_seq(sequence,self.codonprob,self.prob_original))
        # return optimize_single_seq(sequence,self.codonprob,self.prob_original)
        thread_pool = ThreadPoolExecutor(30)
        task_list = []
        for _ in range(self.n_pool-1):
            task_list.append(thread_pool.submit(optimize_single_seq,sequence,self.codonprob,self.prob_original))
        final_seq_list = [sequence]
        for task in as_completed(task_list):
            final_seq_list.append(task.result())
        return final_seq_list


    def select_best(self,sequence, by={"MFE":"min","CAI":"max","CPB":"max","ENC":"min"},
                    homopolymers=0, exclude_motifs = [], top=None): #,"GC":55
        '''
        Sort the pool of generated sequences based on different metrics, and
        output the best ones.

        Parameters
        ----------
        by : {"MFE":"min/max","MFEini":"min/max","CAI":"min/max","CPB":"min/max",
              "ENC":"min/max","GC":float}, default={"MFE":"min","CAI":"max","CPB":"max","ENC":"min"}
            Metrics to use in the evaluation of sequences and whether to
            maximize or minimize. For GC, user specified target float
            
        homopolymers : int, default=0
            Removes sequences containing homopolymer sequences of n repeats.
            
        exclude_motifs : list, default=[]
            Removes sequences including certain motifs. Excessively short 
            motifs are recomended against, since the probability of they 
            appearing by chance is high.
            
        top : int or None, default=None
            Output only the top N sequences. If None, all sorted sequences are
            outputted.

        Returns
        -------
        best = DataFrame
        '''
        relative_codons()
        compute_CPS()
        self.codonprob = self.caculate_codonprob()
        print(time.time())
        optimized_seq_pool =  self.get_optimized_seq_pool(sequence)
        print(time.time())
        
        select_df = pd.DataFrame(index = range(self.n_pool))
        select_df["Sequence"] = optimized_seq_pool
        norm_df = pd.DataFrame(index = range(self.n_pool))
        
        for c in by:
            if c=="MFE":
                # metric = np.array(self.MFE())
                metric = np.array(get_data_in_pool(get_MFE,optimized_seq_pool))
                norm = action(metric,by[c])
            elif c=="MFEini":
                metric = np.array(get_data_in_pool(get_MFEini,optimized_seq_pool))
                norm = action(metric,by[c])
            elif c=="CAI":
                metric = np.array(get_data_in_pool(get_CAI,optimized_seq_pool))
                norm = action(metric,by[c])
            elif c=="CPB":
                metric = np.array(get_data_in_pool(get_CPB,optimized_seq_pool))
                # metric = np.array(self.CPB())
                norm = action(metric,by[c])
            elif c=="ENC":
                metric = np.array(get_data_in_pool(get_ENC,optimized_seq_pool))
                norm = action(metric,by[c])
            elif c=="GC":
                metric = np.array(get_data_in_pool(get_GC,optimized_seq_pool))
                norm = action(np.abs(metric-by[c]),"min")
            else:
                raise TypeError("Invalid 'by' argument.")
            select_df[c] = metric
            # Normalize between 0 and 1
            norm_metric = (norm-np.min(norm))/np.ptp(norm)
            norm_df[c] = norm_metric
        # Create a score based on metrics
        if "GC" not in select_df:
            select_df["GC"] = np.array(get_data_in_pool(get_GC,optimized_seq_pool))
        select_df["Score"] = norm_df.mean(axis=1)
        
        for m in exclude_motifs:
            idx = np.array([m in seq for seq in select_df.Sequence])
            select_df = select_df.loc[np.logical_not(idx),:]
        best = pd.concat([select_df.iloc[:1,],select_df.iloc[1:,].sort_values(by="Score", ascending=False)],ignore_index=True)
        best["Sequence"][0] = "wt:" + best["Sequence"][0]
        return best
    




class TissueOptimizer_Caculator:
    '''
    Optimize object, which contains all required methods for tissue-optimizing 
    the codons an amino-acid or nucleotide sequence. Codons are selected based 
    on the Random Forest features that define tissue-specificities, which
    are considered as the probability of optimizing each codon. Directionality
    is based on tissue-specific SDA ratios. Among the pool of generated 
    sequences, the best ones are selected based several commonly used metrics
    (CAI, ENC, CPB, MFE, etc.).
    
    Parameters
    ----------
    tissue: {"Lung","Breast","Skin","Spleen","Heart","Liver","Salivarygland",
             "Muscle...Skeletal","Tonsil","Smallintestine","Placenta",
             "Appendices","Testis","Rectum","Urinarybladder","Prostate",
             "Esophagus","Kidney","Thyroid","Lymphnode","Artery","Brain",
             "Nerve...Tibial","Gallbladder","Uterus","Pituitary","Colon",
             "Vagina","Duodenum","Fat","Stomach","Adrenal","Fallopiantube",
             "Smoothmuscle","Pancreas","Ovary"}
        Tissue to which optimize the sequence
    
    n_pool: int, default=100
        The number of sequences in the generated pool of optimized sequences.
        
    degree: float between 0-1, default=0.5
        Percentage of codons to optimize. Higher values lead to optimizing 
        all codons. Lower values optimize only the most clearly 
        tissue-specific codons.
    
    prob_original: float between 0-1, default=0.0
        Extent to which original codons are conserved. Higher values lead to 
        more conservative optimizations.
    
    Attributes
    ----------
    pool : list
        Minimum Free Energy of optimized sequences. Predicted secondary 
        structures are based on the Vienna RNA Package (Lorenz et al., 2011).
    
    codonprob : dict
        For each amino acid, it contains a nested dictionary matching each
        codon with (1) its tissue-specific weight and (2) its directionality
        (True if its inclusion is favored in that tissue, False otherwise).
    
    sequence : str
        Original sequence to be optimized.
    
    Examples
    --------
    >>> import custom
    >>> opt = TissueOptimizer("kidney", n_pool=50)
    >>> seq = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSG"
    >>> opt.optimize(seq)
    >>> best_seq = opt.select_best(top=10)
    
    '''
    def __init__(self, tissue, n_pool=1000, degree=0.5, prob_original=0.0):
        if tissue in codon_weights.index:
            self.tissue = tissue
            self.n_pool = n_pool
            self.degree = degree
            self.prob_original = prob_original
            # Get directionality and probabilities for tissue optimization
            # "degree" determines to which extent are unclear codons optimized
        else:
            raise TypeError("Invalid 'tissue' argument. Allowed tissues are: "
                            "Lung, Breast, Skin, Spleen, Heart, Liver, Salivarygland, "
                            "Muscle...Skeletal, Tonsil, Smallintestine, Placenta, "
                            "Appendices, Testis, Rectum, Urinarybladder, Prostate, "
                            "Esophagus, Kidney, Thyroid, Lymphnode, Artery, Brain, "
                            "Nerve...Tibial, Gallbladder, Uterus, Pituitary, Colon, "
                            "Vagina, Duodenum, Fat, Stomach, Adrenal, Fallopiantube, "
                            "Smoothmuscle, Pancreas, Ovary.")
        
    def caculate_codonprob(self):
        threshold = np.percentile(codon_ratios.loc[self.tissue,:].abs(),(1.0-self.degree)*100)
        codonprob = {}
        for aa in set(GENETIC_CODE.values()):
            aa_codons = [c for c in GENETIC_CODE.keys() if np.logical_and((GENETIC_CODE[c] is aa),(c not in ["ATG","TGG","TAA","TGA","TAG"]))]
            # Avoid deterministic behaviour of codons with weight=0
            if any(codon_weights.loc[self.tissue,aa_codons]):
                weight_bkg = codon_weights.loc[self.tissue,aa_codons].mean()
                codweights = {c:codon_weights.loc[self.tissue,c]+weight_bkg for c in aa_codons}
            else:
                codweights = {c:codon_weights.loc[self.tissue,c] for c in aa_codons}
            sumweights = sum(codweights.values())
            if sumweights==0:
                sumweights = 1.0
            coddirection = {c:codon_ratios.loc[self.tissue,c]>0 if np.abs(codon_ratios.loc[self.tissue,c])>threshold else np.nan for c in aa_codons}
            codonprob[aa] = {c:[codweights[c]/sumweights,coddirection[c]] for c in aa_codons}
        # self.codonprob = codonprob
        return codonprob
    

    def get_optimized_seq_pool(self, sequence_list):
        thread_pool = ThreadPoolExecutor(30)
        task_list = []
        for _ in range(self.n_pool-len(sequence_list)):
            task_list.append(thread_pool.submit(optimize_single_seq,np.random.choice(sequence_list),self.codonprob,self.prob_original))
        final_seq_list = [*sequence_list]
        for task in as_completed(task_list):
            final_seq_list.append(task.result())
        return final_seq_list


    def select_best(self,sequence_list,sequence_name_list, by={"MFE":"min","CAI":"max","CPB":"max","ENC":"min"},
                    homopolymers=0, exclude_motifs = [], top=None): #,"GC":55
        '''
        Sort the pool of generated sequences based on different metrics, and
        output the best ones.

        Parameters
        ----------
        by : {"MFE":"min/max","MFEini":"min/max","CAI":"min/max","CPB":"min/max",
              "ENC":"min/max","GC":float}, default={"MFE":"min","CAI":"max","CPB":"max","ENC":"min"}
            Metrics to use in the evaluation of sequences and whether to
            maximize or minimize. For GC, user specified target float
            
        homopolymers : int, default=0
            Removes sequences containing homopolymer sequences of n repeats.
            
        exclude_motifs : list, default=[]
            Removes sequences including certain motifs. Excessively short 
            motifs are recomended against, since the probability of they 
            appearing by chance is high.
            
        top : int or None, default=None
            Output only the top N sequences. If None, all sorted sequences are
            outputted.

        Returns
        -------
        best = DataFrame
        '''
        relative_codons()
        compute_CPS()
        self.codonprob = self.caculate_codonprob()
        optimized_seq_pool =  self.get_optimized_seq_pool(sequence_list)
        
        
        select_df = pd.DataFrame(index = range(self.n_pool))
        select_df["Sequence"] = optimized_seq_pool
        norm_df = pd.DataFrame(index = range(self.n_pool))
        
        for c in by:
            if c=="MFE":
                # metric = np.array(self.MFE())
                metric = np.array(get_data_in_pool(get_MFE,optimized_seq_pool))
                norm = action(metric,by[c])
            elif c=="MFEini":
                metric = np.array(get_data_in_pool(get_MFEini,optimized_seq_pool))
                norm = action(metric,by[c])
            elif c=="CAI":
                metric = np.array(get_data_in_pool(get_CAI,optimized_seq_pool))
                norm = action(metric,by[c])
            elif c=="CPB":
                metric = np.array(get_data_in_pool(get_CPB,optimized_seq_pool))
                # metric = np.array(self.CPB())
                norm = action(metric,by[c])
            elif c=="ENC":
                metric = np.array(get_data_in_pool(get_ENC,optimized_seq_pool))
                norm = action(metric,by[c])
            elif c=="GC":
                metric = np.array(get_data_in_pool(get_GC,optimized_seq_pool))
                norm = action(np.abs(metric-by[c]),"min")
            else:
                raise TypeError("Invalid 'by' argument.")
            select_df[c] = metric
            # Normalize between 0 and 1
            norm_metric = (norm-np.min(norm))/np.ptp(norm)
            norm_df[c] = norm_metric
        # Create a score based on metrics
        if "GC" not in select_df:
            select_df["GC"] = np.array(get_data_in_pool(get_GC,optimized_seq_pool))
        select_df["Score"] = norm_df.mean(axis=1)
        
        # best = pd.concat(select_df.iloc[:1,],select_df.iloc[1:,].sort_values(by="Score", ascending=False),ignore_index=True)
        # best["Sequence"][0] = "wt:" + best["Sequence"]
        best = select_df.iloc[:len(sequence_list),]
        best["SeqName"] = sequence_name_list
        return best
    