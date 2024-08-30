import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def extract_metabolite_smiles(user_input):
    """
    From a reaction SMILE, returns a list of all the reactants and a list of all the products
    """
    x = user_input.split(">>")
    reactants = x[0].split('.')
    products = x[1].split('.')
    return reactants, products

def predict_df(input_path):
    data = pd.read_csv(input_path)
    data['First_Prediction'] = ["3.2.1.8" for i in range(len(data.index)) ]
    data['First_Confidence_score'] = [0.96 for i in range(len(data.index)) ]# Example: doubling the first column
    data['First_Confidence_score_categorical'] = [1 for i in range(len(data.index)) ]
    data['Second_Prediction'] = ["4.5.1.8" for i in range(len(data.index)) ]
    data['Second_Confidence_score'] = [0.22 for i in range(len(data.index)) ]
    data['Second_Confidence_score_categorical'] = [2 for i in range(len(data.index)) ]
    
    first_mean_cs = data.loc[:, 'First_Confidence_score'].mean()
    second_mean_cs = data.loc[:, 'Second_Confidence_score'].mean()
    #EC_number_repartition()
    
    first_frequency_high_cs = len([cs for cs in data['First_Confidence_score'].to_list() if cs > 0.85])*100/len(data.index)
    second_frequency_high_cs = len([cs for cs in data['Second_Confidence_score'].to_list() if cs > 0.85])*100/len(data.index)
    
    data['First_Confidence_score'] = data['First_Confidence_score'].astype('str')
    data['Second_Confidence_score'] = data['Second_Confidence_score'].astype('str')
    
    statistics = {"first_mean_cs":first_mean_cs,"second_mean_cs":second_mean_cs, "first_frequency_high_cs": first_frequency_high_cs, "second_frequency_high_cs":second_frequency_high_cs}
    
    return data, statistics

def from_SMILE_to_mol(reactants, products):
    reactant_list = []
    product_list = []
    for reactant in reactants:
        mol = Chem.MolFromSmiles(reactant)
        img = Draw.MolToImage( mol )
        reactant_list.append( img)

    for product in products:
        mol = Chem.MolFromSmiles(product)
        img = Draw.MolToImage( mol )
        product_list.append( img)
    
    for metabolites in reactant_list:
        if metabolites is None:
            return None
        
    for metabolites in product_list:
        if metabolites is None:
            return None
    return reactant_list, product_list






    