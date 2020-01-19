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