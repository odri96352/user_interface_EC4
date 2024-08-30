import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import pkg_resources
import sklearn
import scipy
import matplotlib.pyplot as plt
import statistics
import re

## predit les 3 premiers ec
## trouve a quelle db ça correspond
## predit le 4e EC

from  models_add_13_new_labels import SmilesClassificationModel
logger = logging.getLogger(__name__)


def sotfmax_raw_outputs(raw_outputs):
      """
      Uses the scipy softmax function on predictions to give a confidence score of every class for one prediction
      Args:
            raw_outputs (2D numpy array): For each prediction, contains list of model outputs
      Return:
            softmax_output (2D numpy array): For each prediction, contains list of score for each possible class
      """
      softmax_output = []
      for prediction in raw_outputs:
            # the scipy softmax function works for 1D numpy arrays
            softmax_output.append(scipy.special.softmax(prediction))
      return softmax_output

def ec_level(ec, level):
    """
    Give an EC number and the wanted level of precision and it returns the EC at said level
    Example: ec_level(1.10.1.1, 3) = 1.10.1
    """
    output = ''
    ec_list = ec.split('.')
    if len(ec_list)> level:
        ec_list = ec_list[:level]
    elif len(ec_list)< level:
        print(f" EC number too small")
        return None
    for number in ec_list:
        output += number
        output += '.'
    return output[:-1]

def ec_class_score(softmaxed_raw_output, predictions):
      """
      For each prediction, returns the score for the class that was predicted.
      Args:
            softmaxed_raw_output (2D numpy array): For each prediction, contains list of score for each possible class calculated by softmaxing the raw_outputs
            predictions (1D numpy array): Class predicted for each EC
      Return:
             (1D numpy array): List of confidence score for the class predicted 
      """
      return [ softmaxed_raw_output[i][predictions[i]] for i in range(len(predictions))]

def predictions_according_to_digit_number(number_of_digits, ec_predicted, true_ec, score_predicted):
      comparison_table = [ec_predicted[i] for i in range(len(ec_predicted)) if ec_level(ec_predicted[i], number_of_digits)==ec_level(true_ec[i], number_of_digits)]
      confidence_score = [score_predicted[i] for i in range(len(ec_predicted)) if ec_level(ec_predicted[i], number_of_digits)==ec_level(true_ec[i], number_of_digits)]
      print(f"Number of predictions with equal first {number_of_digits} digits: {len(comparison_table)}, Frequency: {len(comparison_table)*100/len(ec_predicted)}%")
      print(f"Maximal confidence score: {max(confidence_score)},  minimal confidence score:  {min(confidence_score)}, mean confidence score: {statistics.mean(confidence_score)}")
      return

def evaluate_model(predicted_ec_numbers,ec_confidence_score, y_true, num_max_digit):
      for i in range(1,num_max_digit+1):
            predictions_according_to_digit_number(i, predicted_ec_numbers, y_true, ec_confidence_score)
            print('\n')
      return


def output_df_first_model(predicted_ec, confidence_score, true_ec):
      outputdf = pd.DataFrame({"true_EC_number": true_ec,
                            "predicted_EC_number": predicted_ec,
                            "confidence_score": confidence_score})
      return outputdf

def output_df_second_model(predicted_ec, confidence_score, true_ec, model_num):
      outputdf = pd.DataFrame({"true_EC_number": true_ec,
                            "predicted_EC_number": predicted_ec,
                            "confidence_score": confidence_score,
                            "model_num": model_num})
      return outputdf

def reorder_lists(first_list, correct_order):
    combined = zip(correct_order, first_list)
    sorted_combined = sorted(combined)
    result = [element for _, element in sorted_combined]
    return result


if __name__== '__main__':
    #############################
    # Test df
    input = pd.read_csv("models/test_dataset/expanded_testing_df_2.csv")
    input = input.head(1000)
    print("head", input.head())
    print("lenght", len(input.index))

    true_ec = input.ec_num.values.tolist()
    rxn_smile = input.rxn.values.tolist()

    
    
    def predict_full_pipeline(true_ec,rxn_smile):
        convert_ec_to_model_num = pd.read_csv("models/count_fourth_ec_group_by_three_first_ec_and_database_repartition.csv")
        #############################
        # Import the trained model to predict the three first EC number

        # first_model = SmilesClassificationModel("bert", "/home/audrey/Documents/Audrey_INSA_internship/personal_gitub/InsaInternship/BECPred/collab_test/model_full_enzymemap_n_recon3d_data_augmentation_with_h_no_rdkit", use_cuda=torch.cuda.is_available())
        # first_index_df =  pd.read_csv('/home/audrey/Documents/Audrey_INSA_internship/personal_gitub/InsaInternship/BECPred/collab_test/model_full_enzymemap_n_recon3d_data_augmentation_with_h_no_rdkit/all_labels.csv')
        
        # first_model = SmilesClassificationModel("bert", "/home/audrey/Documents/Audrey_INSA_internship/personal_gitub/InsaInternship/BECPred/full_enzymemap_and_thg_data_augmentation/with_rdkit/without_adding_a_hydrogen/model/checkpoint-386000", use_cuda=torch.cuda.is_available())
        # first_index_df =  pd.read_csv('/home/audrey/Documents/Audrey_INSA_internship/personal_gitub/InsaInternship/BECPred/full_enzymemap_and_thg_data_augmentation/with_rdkit/without_adding_a_hydrogen/index_all_labels.csv')
        
        first_model = SmilesClassificationModel("bert", "models/model_full_enzymemap_n_recon3d_data_augmentation_with_h_no_rdkit_trained_on_collab", num_labels=325, use_cuda=torch.cuda.is_available())
        first_index_df =  pd.read_csv('models/model_full_enzymemap_n_recon3d_data_augmentation_with_h_no_rdkit_trained_on_collab/all_labels.csv')
        

        # first_model = SmilesClassificationModel("bert", "/home/audrey/Documents/Audrey_INSA_internship/BECPred_github/BEC-Pred/BECPred/model", use_cuda=torch.cuda.is_available())
        # first_index_df =  pd.read_csv('/home/audrey/Documents/Audrey_INSA_internship/BECPred_github/BEC-Pred/data/index_to_ec_number.csv')

        ##############################
        #Load all the 16 models
        label_list = [723, 637, 692,753,640,597,581,709, 577,699, 699, 703,751, 717, 736, 737, 366]
        model_0 = SmilesClassificationModel("bert", "models/epoch_15_df_0" , use_cuda=torch.cuda.is_available(), num_labels=label_list[0])
        model_1 = SmilesClassificationModel("bert","models/epoch_3_df_1" , use_cuda=torch.cuda.is_available(), num_labels=label_list[1])
        model_2 = SmilesClassificationModel("bert","models/epoch_3_df_2" , use_cuda=torch.cuda.is_available(), num_labels=label_list[2])
        model_3 = SmilesClassificationModel("bert", "models/epoch_4_df_3" , use_cuda=torch.cuda.is_available(), num_labels=label_list[3])
        model_4 = SmilesClassificationModel("bert", "models/epoch_5_df_4" , use_cuda=torch.cuda.is_available(), num_labels=label_list[4])
        model_5 = SmilesClassificationModel("bert", "models/epoch_5_df_5" , use_cuda=torch.cuda.is_available(), num_labels=label_list[5])
        model_6 = SmilesClassificationModel("bert", "models/epoch_5_df_6" , use_cuda=torch.cuda.is_available(), num_labels=label_list[6])
        model_7 = SmilesClassificationModel("bert", "models/epoch_5_df_7" , use_cuda=torch.cuda.is_available(), num_labels=label_list[7])
        model_8 = SmilesClassificationModel("bert", "models/epoch_5_df_8" , use_cuda=torch.cuda.is_available(), num_labels=label_list[8])
        model_9 = SmilesClassificationModel("bert", "models/epoch_5_df_9" , use_cuda=torch.cuda.is_available(), num_labels=label_list[9])
        model_10 = SmilesClassificationModel("bert", "models/epoch_5_df_10" , use_cuda=torch.cuda.is_available(), num_labels=label_list[10])
        model_11 = SmilesClassificationModel("bert", "models/epoch_5_df_11" , use_cuda=torch.cuda.is_available(), num_labels=label_list[11])
        model_12 = SmilesClassificationModel("bert", "models/epoch_5_df_12" , use_cuda=torch.cuda.is_available(), num_labels=label_list[12])
        model_13 = SmilesClassificationModel("bert", "models/epoch_5_df_13" , use_cuda=torch.cuda.is_available(), num_labels=label_list[13])
        model_14 = SmilesClassificationModel("bert", "models/epoch_5_df_14" , use_cuda=torch.cuda.is_available(), num_labels=label_list[14])
        model_15 = SmilesClassificationModel("bert", "models/epoch_5_df_15" , use_cuda=torch.cuda.is_available(), num_labels=label_list[15])
        model_16 = SmilesClassificationModel("bert", "models/epoch_5_df_16" , use_cuda=torch.cuda.is_available(), num_labels=label_list[16])

        model_list = [model_0,model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8,model_9,model_10,model_11,model_12,model_13,model_14,model_15,model_16]
        ##############################
        # Load all the 16 indexes
        index_folder_path = "models/index_folder/"
        index_list = []
        dir_list = os.listdir(index_folder_path)
        index_list = []
        filename = []

        def extract_number(file_name):
            match = re.search(r'(\d+)', file_name)
            return int(match.group(1)) if match else 0

        for x in os.listdir(index_folder_path):
            if x.startswith("index_all_label_df_"):
                filename.append(x)

        sorted_file_names = sorted(filename, key=extract_number)        
        index_list = [pd.read_csv(index_folder_path + i) for i in sorted_file_names]
        print("index_list", index_list)
        ##############################
        # First model
        print('First model')
        print('Length of input:', len(rxn_smile))
        first_pred, first_raw_outputs = first_model.predict(rxn_smile)

        three_lvl_ec_list = [first_index_df._get_value(int(prediction), 'ec_number') for prediction in first_pred]

        ec_confidence_score = ec_class_score(sotfmax_raw_outputs(first_raw_outputs), first_pred)
        
        print("three_lvl_ec", three_lvl_ec_list)

        evaluate_model(three_lvl_ec_list, ec_confidence_score, true_ec, 3)
        output = output_df_first_model(three_lvl_ec_list, ec_confidence_score, true_ec)
        ###############################
        # Second model
        second_pred_list = []
        second_true_ec_list = []
        ec_confidence_score_list = []
        four_lvl_ec = []
        model_number_list = []
        correct_order = []


        ### Inefficient method
        # for i in range(len(three_lvl_ec_list)):
        #     three_lvl_ec = three_lvl_ec_list[i]
        #     model_num = convert_ec_to_model_num[convert_ec_to_model_num["Three_first_ec"] == three_lvl_ec].iloc[0]['Database_repartition']
        #     model_number_list.append(model_num)
        #     second_model = model_list[model_num]
        #     second_index_df = index_list[model_num]
        #     second_pred, second_raw_outputs = second_model.predict([rxn_smile[i]])
        #     second_pred_list.append(second_pred[0])
        #     second_raw_outputs_list.append(second_raw_outputs[0])
        #     four_lvl_ec.append(second_index_df._get_value(int(second_pred[0]), 'ec_number'))
        # print("three_lvl_ec", three_lvl_ec)
        # print("model_num", model_num)
        # print("second_model", second_model)
        # print("second_index_df", second_index_df) 
        # print("rxn_smile[i]", rxn_smile[i])
        # print("second_pred", second_pred[0])
        # print("second_raw_outputs", second_raw_outputs)
        # print("four_lvl_ec", four_lvl_ec)

        ## Save each reaction SMILE according to which model they will be given as input
        model_number_dic = {0:{"rxn":[], "true_ec":[], "index":[]},1:{"rxn":[], "true_ec":[], "index":[]},2:{"rxn":[], "true_ec":[], "index":[]},3:{"rxn":[], "true_ec":[], "index":[]}, 4:{"rxn":[], "true_ec":[], "index":[]},5:{"rxn":[], "true_ec":[], "index":[]},6:{"rxn":[], "true_ec":[], "index":[]},7:{"rxn":[], "true_ec":[], "index":[]},8:{"rxn":[], "true_ec":[], "index":[]},9:{"rxn":[], "true_ec":[], "index":[]},10:{"rxn":[], "true_ec":[], "index":[]},11:{"rxn":[], "true_ec":[], "index":[]},12:{"rxn":[], "true_ec":[], "index":[]},13:{"rxn":[], "true_ec":[], "index":[]},14:{"rxn":[], "true_ec":[], "index":[]},15:{"rxn":[], "true_ec":[], "index":[]},16:{"rxn":[], "true_ec":[], "index":[]}}
        for i in range(len(three_lvl_ec_list)):
            three_lvl_ec = three_lvl_ec_list[i]
            ## Save each reaction SMILE with it's index to put them back in the initial order after the prediction
            index = i
            try:
                model_num = convert_ec_to_model_num[convert_ec_to_model_num["Three_first_ec"] == three_lvl_ec].iloc[0]['Database_repartition']
            except IndexError:
                print(f"{three_lvl_ec} isn't a valid EC prediction")
            model_number_dic[model_num]["rxn"].append([rxn_smile[i]])
            model_number_dic[model_num]["true_ec"].append(true_ec[i])
            model_number_dic[model_num]["index"].append(index)

        #print("model_number_dic", model_number_dic)

        ## Go through all the 17 models and predict EC numbers of the reaction SMILEs.
        for j in range(17):
            model_num = j
            if len(model_number_dic[model_num]["rxn"])>0:
                second_model = model_list[model_num]
                second_index_df = index_list[model_num]

                second_pred, second_raw_outputs = second_model.predict(model_number_dic[model_num]["rxn"])

                model_number_list += [model_num for i in range(len(second_pred))]
                second_pred_list += second_pred
                
                ec_confidence_score = ec_class_score(sotfmax_raw_outputs(second_raw_outputs), second_pred)
                ec_confidence_score_list += ec_confidence_score
                
                second_true_ec_list += model_number_dic[model_num]["true_ec"]
                
                four_lvl_ec += [second_index_df._get_value(int(pred), 'ec_number') for pred in second_pred]

                
                correct_order += model_number_dic[model_num]["index"]
                
                # print("second_pred_list", second_pred_list)
                # print("ec_confidence_score_list", ec_confidence_score_list)
                # print("second_true_ec_list", second_true_ec_list)
                # print("four_lvl_ec", four_lvl_ec)

        ## Put the predicted EC back in order
        four_lvl_ec = reorder_lists(four_lvl_ec, correct_order)
        second_true_ec_list = reorder_lists(second_true_ec_list, correct_order)
        ec_confidence_score_list = reorder_lists(ec_confidence_score_list, correct_order)
        model_number_list = reorder_lists(model_number_list, correct_order)




        #ec_confidence_score = ec_class_score(sotfmax_raw_outputs(second_raw_outputs_list), second_pred_list)
        evaluate_model(four_lvl_ec, ec_confidence_score_list, second_true_ec_list, 4)
        output = output_df_second_model(four_lvl_ec, ec_confidence_score_list, second_true_ec_list, model_number_list)


    def one_prediction_full_pipeline(rxn_smile):
        convert_ec_to_model_num = pd.read_csv("models/count_fourth_ec_group_by_three_first_ec_and_database_repartition.csv")
        #############################
        # Import the trained model to predict the three first EC number

        # first_model = SmilesClassificationModel("bert", "/home/audrey/Documents/Audrey_INSA_internship/personal_gitub/InsaInternship/BECPred/collab_test/model_full_enzymemap_n_recon3d_data_augmentation_with_h_no_rdkit", use_cuda=torch.cuda.is_available())
        # first_index_df =  pd.read_csv('/home/audrey/Documents/Audrey_INSA_internship/personal_gitub/InsaInternship/BECPred/collab_test/model_full_enzymemap_n_recon3d_data_augmentation_with_h_no_rdkit/all_labels.csv')
        
        # first_model = SmilesClassificationModel("bert", "/home/audrey/Documents/Audrey_INSA_internship/personal_gitub/InsaInternship/BECPred/full_enzymemap_and_thg_data_augmentation/with_rdkit/without_adding_a_hydrogen/model/checkpoint-386000", use_cuda=torch.cuda.is_available())
        # first_index_df =  pd.read_csv('/home/audrey/Documents/Audrey_INSA_internship/personal_gitub/InsaInternship/BECPred/full_enzymemap_and_thg_data_augmentation/with_rdkit/without_adding_a_hydrogen/index_all_labels.csv')
        
        first_model = SmilesClassificationModel("bert", "models/model_full_enzymemap_n_recon3d_data_augmentation_with_h_no_rdkit_trained_on_collab", num_labels=325, use_cuda=torch.cuda.is_available())
        first_index_df =  pd.read_csv('models/model_full_enzymemap_n_recon3d_data_augmentation_with_h_no_rdkit_trained_on_collab/all_labels.csv')
        

        # first_model = SmilesClassificationModel("bert", "/home/audrey/Documents/Audrey_INSA_internship/BECPred_github/BEC-Pred/BECPred/model", use_cuda=torch.cuda.is_available())
        # first_index_df =  pd.read_csv('/home/audrey/Documents/Audrey_INSA_internship/BECPred_github/BEC-Pred/data/index_to_ec_number.csv')

        ##############################
        #Load all the 16 models
        label_list = [723, 637, 692,753,640,597,581,709, 577,699, 699, 703,751, 717, 736, 737, 366]
        model_0 = SmilesClassificationModel("bert", "models/epoch_15_df_0" , use_cuda=torch.cuda.is_available(), num_labels=label_list[0])
        model_1 = SmilesClassificationModel("bert","models/epoch_3_df_1" , use_cuda=torch.cuda.is_available(), num_labels=label_list[1])
        model_2 = SmilesClassificationModel("bert","models/epoch_3_df_2" , use_cuda=torch.cuda.is_available(), num_labels=label_list[2])
        model_3 = SmilesClassificationModel("bert", "models/epoch_4_df_3" , use_cuda=torch.cuda.is_available(), num_labels=label_list[3])
        model_4 = SmilesClassificationModel("bert", "models/epoch_5_df_4" , use_cuda=torch.cuda.is_available(), num_labels=label_list[4])
        model_5 = SmilesClassificationModel("bert", "models/epoch_5_df_5" , use_cuda=torch.cuda.is_available(), num_labels=label_list[5])
        model_6 = SmilesClassificationModel("bert", "models/epoch_5_df_6" , use_cuda=torch.cuda.is_available(), num_labels=label_list[6])
        model_7 = SmilesClassificationModel("bert", "models/epoch_5_df_7" , use_cuda=torch.cuda.is_available(), num_labels=label_list[7])
        model_8 = SmilesClassificationModel("bert", "models/epoch_5_df_8" , use_cuda=torch.cuda.is_available(), num_labels=label_list[8])
        model_9 = SmilesClassificationModel("bert", "models/epoch_5_df_9" , use_cuda=torch.cuda.is_available(), num_labels=label_list[9])
        model_10 = SmilesClassificationModel("bert", "models/epoch_5_df_10" , use_cuda=torch.cuda.is_available(), num_labels=label_list[10])
        model_11 = SmilesClassificationModel("bert", "models/epoch_5_df_11" , use_cuda=torch.cuda.is_available(), num_labels=label_list[11])
        model_12 = SmilesClassificationModel("bert", "models/epoch_5_df_12" , use_cuda=torch.cuda.is_available(), num_labels=label_list[12])
        model_13 = SmilesClassificationModel("bert", "models/epoch_5_df_13" , use_cuda=torch.cuda.is_available(), num_labels=label_list[13])
        model_14 = SmilesClassificationModel("bert", "models/epoch_5_df_14" , use_cuda=torch.cuda.is_available(), num_labels=label_list[14])
        model_15 = SmilesClassificationModel("bert", "models/epoch_5_df_15" , use_cuda=torch.cuda.is_available(), num_labels=label_list[15])
        model_16 = SmilesClassificationModel("bert", "models/epoch_5_df_16" , use_cuda=torch.cuda.is_available(), num_labels=label_list[16])

        model_list = [model_0,model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8,model_9,model_10,model_11,model_12,model_13,model_14,model_15,model_16]
        ##############################
        # Load all the 16 indexes
        index_folder_path = "models/index_folder/"
        index_list = []
        dir_list = os.listdir(index_folder_path)
        index_list = []
        filename = []

        def extract_number(file_name):
            match = re.search(r'(\d+)', file_name)
            return int(match.group(1)) if match else 0

        for x in os.listdir(index_folder_path):
            if x.startswith("index_all_label_df_"):
                filename.append(x)

        sorted_file_names = sorted(filename, key=extract_number)        
        index_list = [pd.read_csv(index_folder_path + i) for i in sorted_file_names]
        print("index_list", index_list)
        ##############################
        # First model
        print('First model')
        print('Length of input:', len(rxn_smile))
        first_pred, first_raw_outputs = first_model.predict(rxn_smile)

        three_lvl_ec_list = [first_index_df._get_value(int(prediction), 'ec_number') for prediction in first_pred]

        ec_confidence_score = ec_class_score(sotfmax_raw_outputs(first_raw_outputs), first_pred)
        
        print("three_lvl_ec", three_lvl_ec_list)


        ###############################
        # Second model
        four_lvl_ec = []
        model_number_list = []

        ## Save each reaction SMILE according to which model they will be given as input
        model_number_dic = {0:{"rxn":[]},1:{"rxn":[] },2:{"rxn":[] },3:{"rxn":[] }, 4:{"rxn":[] },5:{"rxn":[] },6:{"rxn":[] },7:{"rxn":[] },8:{"rxn":[] },9:{"rxn":[] },10:{"rxn":[] },11:{"rxn":[] },12:{"rxn":[] },13:{"rxn":[] },14:{"rxn":[] },15:{"rxn":[] },16:{"rxn":[] }}
        for i in range(len(three_lvl_ec_list)):
            three_lvl_ec = three_lvl_ec_list[i]
            ## Save each reaction SMILE with it's index to put them back in the initial order after the prediction
            index = i
            try:
                model_num = convert_ec_to_model_num[convert_ec_to_model_num["Three_first_ec"] == three_lvl_ec].iloc[0]['Database_repartition']
            except IndexError:
                print(f"{three_lvl_ec} isn't a valid EC prediction")
            model_number_dic[model_num]["rxn"].append([rxn_smile[i]])

        ## Go through all the 17 models and predict EC numbers of the reaction SMILEs.
        for j in range(17):
            model_num = j
            if len(model_number_dic[model_num]["rxn"])>0:
                second_model = model_list[model_num]
                second_index_df = index_list[model_num]

                second_pred, second_raw_outputs = second_model.predict(model_number_dic[model_num]["rxn"])

                model_number_list += [model_num for i in range(len(second_pred))]

                ec_confidence_score = ec_class_score(sotfmax_raw_outputs(second_raw_outputs), second_pred)
                
                four_lvl_ec += [second_index_df._get_value(int(pred), 'ec_number') for pred in second_pred]
        return four_lvl_ec[0], ec_confidence_score

    #predict_full_pipeline(true_ec,rxn_smile)
    print("one_prediction_full_pipeline: ",one_prediction_full_pipeline(["C(C)C.O=P(OP(O)(OC[C@@H]1[C@@H](O)[C@@H](O)[C@H](n2cnc3c(N)ncnc32)O1)=O)(OC[C@H]1O[C@@H](N2C=CCC(C(=O)N)=C2)[C@H](O)[C@@H]1O)O.O=O>>C(C)CO.OC(C)C.O=P(OP(O)(=O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O)(OC[C@@H]1[C@@H](O)[C@@H](O)[C@H]([n+]2cccc(C(=O)N)c2)O1)O.O"]))
    # correct EC: 1.14.13.25

