
"""
@author: Preston Phillips
"""
# importing the analysis part from the FareTipPredictionClass file
from FareTipPredictionClass import FareTipPrediction

# loading the data into the workspace
fareTip = FareTipPrediction('D:/Vaishnavi_Acads/MIT/Fall 2017/Python/Final/Projectcode/Data/final.csv')

#variable for the logical mapping of the payment type 
cleanup_paymentType = {"payment_type": 
            {"CRD" : "CARD", "CREDIT" : "CARD", "Cre" : "CARD", 1 : "CARD","Credit" : "CARD","1":"CARD" ,
             "CSH" : "CASH", "CAS" : "CASH" ,2 : "CASH", "Cash" : "CASH"}}
    
# calling the datacleanup method
fareTip.dataCleanup(cleanup_paymentType)
# Visualization for the MTA TAX Vs Number of trips
fareTip.plotGraph('MTA Tax','Number of Trips','Plot for MTA TAX', 'mta_tax','bar')
# Visualization for the Passenger count Vs Number of trips
fareTip.plotGraph('Passenger count','Number of Trips','Plot for Passenger Count', 'passenger_count','bar')
# Visualization for the Payment Type Vs Number of trips
fareTip.plotGraph('Payment Type','Number of Trips','Plot for Payment Mode', 'payment_type','bar')
# calling the function for outlier removal and visualization on the density of pick up and drop off
fareTip.fareDataAnalysis()
# calling the function for model creation for the prediction of the tip percentage
fareTip.modelTipPercentagePrediction()
# Distribution of the tip percentage over transcation
fareTip.plotGraph('Tip (%)','Number of transcations', 'Distribution of Tip (%) transactions', 'tip_percentage', 'line')

#"Tip_percentage" showed that 60% of all transactions did not give tip 
# A second tip at 40% corresponds to the usual NYC customary gratuity rate which fluctuates between 10% and 40%
