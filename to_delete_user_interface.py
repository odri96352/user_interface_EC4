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

    # reactant1 = Chem.MolFromSmiles("Cc1ccccc1")
    # reactant2 = Chem.MolFromSmiles('Cc1ccccc1')
    # product1 = Chem.MolFromSmiles('Cc1ccccc1')
    # product2 = Chem.MolFromSmiles('Cc1ccccc1')
    
    # met_list = [reactant1, reactant2, product1, product2]
    
    for metabolites in reactant_list:
        if metabolites is None:
            return None
        
    for metabolites in product_list:
        if metabolites is None:
            return None
        
    # print("reactant1", reactant1)
    # print("reactant2", reactant2)
    # print("product1", product1)
    # print("product2", product2)
    
    # img1 = Draw.MolToImage(reactant1)
    # img2 = Draw.MolToImage(reactant2)
    # img3 = Draw.MolToImage(product1)
    # img4 = Draw.MolToImage(product2)

    # reactant_imgs = [img1,img2 ]
    # product_imgs = [img3, img4]
    # replace with reactants and products

    return reactant_list, product_list

print(extract_metabolite_smiles("C(C)C.O=P(OP(O)(OC[C@@H]1[C@@H](O)[C@@H](O)[C@H](n2cnc3c(N)ncnc32)O1)=O)(OC[C@H]1O[C@@H](N2C=CCC(C(=O)N)=C2)[C@H](O)[C@@H]1O)O.O=O>>C(C)CO.OC(C)C.O=P(OP(O)(=O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O)(OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n+]2cccc(C(=O)N)c2)O1)O.O"))

reactants, products = extract_metabolite_smiles("C(C)C.O=P(OP(O)(OC[C@@H]1[C@@H](O)[C@@H](O)[C@H](n2cnc3c(N)ncnc32)O1)=O)(OC[C@H]1O[C@@H](N2C=CCC(C(=O)N)=C2)[C@H](O)[C@@H]1O)O.O=O>>C(C)CO.OC(C)C.O=P(OP(O)(=O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O)(OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n+]2cccc(C(=O)N)c2)O1)O.O")

reactants_img, products_img = from_SMILE_to_mol(reactants, products)

for r in reactants_img:
    r.show()

for p in products_img:
    p.show()