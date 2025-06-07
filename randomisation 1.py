from PyQt6 import uic
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
import random

app = QApplication([])
window = uic.loadUi("experiment_ui.ui")
window.pageContainer.setCurrentIndex(0)  #Always start from the first page (consent)
window.totalPages = len(window.pageContainer.children()) - 1

result_file = "result_file.csv"
file = open("result_file.csv", "a") #Create a file for storing results
file.close() #Close in time to avoid data leak

#Useful constants:
experiment_state = {
    "participant_id": None,
    "trial_count": 0, #Number of trials that participant completes
    "max_trials": 5,  #Maximum number of trials per participant
    "condition": None
}

debriefing_page_index = window.totalPages - 2

#Consent validation
def validateConsent():
    if not window.ConsentCheckbox.isChecked():
        set_message(window.lblMessageConsent, "You must provide your consent to proceed.", "red")
        return False
    window.lblMessageConsent.setText("")
    experiment_state["participant_id"] = get_new_participant_id(result_file)
    return True

window.ConsentCheckbox.stateChanged.connect(validateConsent)

def get_new_participant_id(file_path=result_file):
    max_participant_id = 1
    participant_index = 0 #Participant column index
    file = open("result_file.csv", "r")
    if file_path:
        lines = file.readlines()
        if lines:
            last_line = lines[-1].strip()
            split_line = last_line.split(",")
            participant_id_no = split_line[participant_index].replace("participant_", "")  #Extract the ID number
            max_participant_id = int(participant_id_no) + 1  #Add 1 to the last ID
        file.close()
        return f"participant_{max_participant_id}"

#Demographics validation
def validateDemographics():
    if window.spinAge.value() < 18:
        set_message(window.lblMessageDemo, "Please enter your age. Please note that individuals under 18 are not eligible to participate in this study.", "red")
        return False
    if not (window.Rdbmale.isChecked() or window.Rdbfemale.isChecked() or window.Rdbother.isChecked()):
        set_message(window.lblMessageDemo, "Please select your gender.", "red")
        return False
    if window.EduBox.currentIndex() == 0:
        set_message(window.lblMessageDemo, "Please select your education level.", "red")
        return False
    window.lblMessageDemo.setText("")
    return True

#Condition assignment & Randomisation
condition = {
    "control": "control",
    "presentation": {
        "point+interval": [80, 90],
        "interval": [80, 90],
    }
}

def load_condition_counts(file_path, condition):
    condition_counts = {"control": 0, "80 point+interval": 0, "80 interval": 0, "90 point+interval": 0, "90 interval": 0}  # Initialize the counts for valid conditions
    condition_index = 24 #Condition column index
    file = open(file_path, "r")
    if file_path:
        lines = file.readlines()
        for row in lines[1:]: #Skip the header line and track counts of each condition
            split_row = row.strip().split(",")
            if len(split_row) >= 6:
                condition = int(split_row[condition_index].strip())
                if condition in condition_counts:
                    condition_counts[condition] += 1
    file.close()
    return condition_counts

def assign_condition(design_type, condition_counts, urn_types, trial_count):
    min_count = min(condition_counts.values())  #Find the minimum count of the condition (i.e., fewest trials taken)
    eligible_conditions = []
    for urn in condition_counts:
        count = condition_counts[urn]
        if count == min_count:
            eligible_conditions.append(urn)
    if design_type == "between" and trial_count == 0: #For between design, only count for the first trial
        assigned_condition = random.choice(eligible_conditions)
        condition_counts[assigned_condition] += 1
    return assigned_condition, condition_counts

