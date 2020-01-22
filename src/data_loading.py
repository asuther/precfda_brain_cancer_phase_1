import pandas as pd
import constants

def load_sc1_data():
    sc1_Phase1_GE_FeatureMatrix = pd.read_csv(f'{constants.base_dir}/data/raw/sc1_Phase1_GE_FeatureMatrix.tsv', sep='\t')\
        .set_index('PATIENTID')
    sc1_Phase1_GE_Outcome = pd.read_csv(f'{constants.base_dir}/data/raw/sc1_Phase1_GE_Outcome.tsv', sep='\t')\
        .set_index('PATIENTID')
    sc1_Phase1_GE_Phenotype = pd.read_csv(f'{constants.base_dir}/data/raw/sc1_Phase1_GE_Phenotype.tsv', sep='\t')\
        .set_index('PATIENTID')

    return sc1_Phase1_GE_FeatureMatrix, sc1_Phase1_GE_Outcome['SURVIVAL_STATUS'], sc1_Phase1_GE_Phenotype

def load_sc2_data():
    sc2_Phase1_CN_FeatureMatrix = pd.read_csv(f'{constants.base_dir}/data/raw/sc2_Phase1_CN_FeatureMatrix.tsv', sep    ='\t')\
        .set_index('PATIENTID')
    sc2_Phase1_CN_Outcome = pd.read_csv(f'{constants.base_dir}/data/raw/sc2_Phase1_CN_Outcome.tsv', sep='\t')\
        .set_index('PATIENTID')
    sc2_Phase1_CN_Phenotype = pd.read_csv(f'{constants.base_dir}/data/raw/sc2_Phase1_CN_Phenotype.tsv', sep='\t')\
        .set_index('PATIENTID')
    
    return sc2_Phase1_CN_FeatureMatrix, sc2_Phase1_CN_Outcome['SURVIVAL_STATUS'], sc2_Phase1_CN_Phenotype


def load_sc3_data():
    sc3_Phase1_CN_GE_FeatureMatrix = pd.read_csv(f'{constants.base_dir}/data/raw/sc3_Phase1_CN_GE_FeatureMatrix.tsv',sep='\t')\
        .set_index('PATIENTID')
    sc3_Phase1_CN_GE_Outcome = pd.read_csv(f'{constants.base_dir}/data/raw/sc3_Phase1_CN_GE_Outcome.tsv', sep='\t') \
        .set_index('PATIENTID')
    sc3_Phase1_CN_GE_Phenotype = pd.read_csv(f'{constants.base_dir}/data/raw/sc3_Phase1_CN_GE_Phenotype.tsv', sep='\t') \
        .set_index('PATIENTID')

    return sc3_Phase1_CN_GE_FeatureMatrix, sc3_Phase1_CN_GE_Outcome['SURVIVAL_STATUS'], sc3_Phase1_CN_GE_Phenotype
