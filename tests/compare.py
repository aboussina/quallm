import json
import pickle
import pandas as pd
from datetime import datetime, timedelta
import glob
from sklearn.metrics import classification_report

def compare_answers(ground_truth, predictions, flexible_window_minutes=10):

    patient_accuracies = {}
    mismatches = {}

    yes_no_check = [1, 2, 3, 4, 7, 8, 11, 14, 17, 18, 22, 25, 31, 34, 35, 37, 40]
    integer_check = [9, 21, 28, 36] 
    time_check = [6, 13, 16, 20, 24, 27, 30, 33, 39, 42]
    total_patients = 0
    correct_dispositions = 0
    y_pred =[]
    y = []

    for patient_id, pred_data in predictions.items():
        pred_answers = pred_data['answers']
        gt_answers = ground_truth[patient_id]
        mismatches[patient_id] = {}
        
        total_questions = 0
        correct_answers = 0
        total_patients +=1
        
        gt_category = gt_answers['category']
        pred_category = pred_data['category']
        
        y.append(gt_category)
        y_pred.append(pred_category)

        if str(gt_category) == str(pred_category):
            correct_dispositions += 1
        else:
            mismatches[patient_id]['category'] = {
                'ground_truth': gt_category, 
                'prediction': pred_category
            }

        for q, pred_answer in pred_answers.items():
            q_num = int(q[1:])
            q = 'q'+q[1:]
            if gt_answers[q]:  # Compare only if ground truth answer exists
                gt_answer = gt_answers[q]

                if q_num in time_check + [10]:
                    pred_answer = pred_answer[:5]
                    gt_time = datetime.strptime(gt_answer, '%H:%M')
                    pred_time = datetime.strptime(pred_answer, '%H:%M')
                    time_difference = abs((gt_time - pred_time).total_seconds() / 60)
                    if time_difference <= flexible_window_minutes:
                        pred_answer = gt_answer  # If within window, consider it a match
                    
                total_questions += 1
                if str(gt_answer) == str(pred_answer):
                    correct_answers += 1
                else:
                    mismatches[patient_id][q] = {'ground_truth': gt_answer, 'prediction': pred_answer}
        
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        patient_accuracies[patient_id] = accuracy

    cr = classification_report(y, y_pred)
    # Remove patient_ids with no mismatches
    mismatches = {k: v for k, v in mismatches.items() if v}
    print("Disposition Accuracy: ", correct_dispositions / total_patients)

    return patient_accuracies, mismatches, cr


predictions = {}
result_files = glob.glob("./outputs/*")
for rf in result_files:
    if ".json" in rf:
        har = rf.split('\\')[1].split('.')[0]
        with open(rf) as f:
            result = json.load(f)

            # Aggregate across encounters with a basic heuristic
            categories = {k: v['category'] for k, v in result.items()}
            passes = [k for k, v in categories.items() if v == 'E']
            fails = [k for k, v in categories.items() if v == 'D']
            not_in_measure = [k for k, v in categories.items() if v == 'B']
            excluded = [k for k, v in categories.items() if v == 'X']

            if passes:
                predictions[har] = result[passes[0]]
            elif fails:
                predictions[har] = result[fails[0]]
            elif not_in_measure:
                predictions[har] = result[not_in_measure[0]]
            elif excluded:
                predictions[har] = result[excluded[0]]

with open(r'./inputs/ground_truth.json', 'r') as j: 
    ground_truth = json.loads(j.read())

accuracy, mismatches, report = compare_answers(ground_truth, predictions)

print(f"Accuracy: {accuracy}")
print("Mismatches:", json.dumps(mismatches, indent=4))
print(report)
json.dump(mismatches, open("mismatches.json", 'w'), indent=4)